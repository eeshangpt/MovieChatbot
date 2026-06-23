"""Smoke tests using monkeypatched model loaders -- no network/model
downloads required. Validates wiring, validation logic, and error handling.
"""

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from relevance_gate.config import RelevanceClassifierConfig, Tier1Config, Tier2Config
from relevance_gate.data_generation import (
    DEFAULT_POSITIVE_CATEGORIES,
    generate_synthetic_dataset,
)
from relevance_gate.embedding_classifier import EmbeddingLRClassifier
from relevance_gate.exceptions import (
    ConfigError,
    InvalidInputError,
    ModelNotFittedError,
)
from relevance_gate.relevance_classifier import QueryRelevanceClassifier
from relevance_gate.schemas import Label
from relevance_gate.tier1_ensemble import Tier1Ensemble


def test_config_validation():
    try:
        Tier1Config(relevant_threshold=0.3, out_of_domain_threshold=0.5)
        raise AssertionError("Expected ConfigError")
    except ConfigError:
        pass

    try:
        Tier1Config(w_lr=0.0, w_ft=0.0)
        raise AssertionError("Expected ConfigError")
    except ConfigError:
        pass

    try:
        RelevanceClassifierConfig(default_policy_label="maybe")
        raise AssertionError("Expected ConfigError")
    except ConfigError:
        pass

    print("OK: config validation rejects bad thresholds/weights/policy")


def make_fake_embedder(dim=16):
    class FakeEmbedder:
        def encode(
            self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        ):
            # Deterministic pseudo-embeddings: push "movie"/"film" texts into
            # a separate cluster so the LR head has a learnable signal.
            out = []
            for t in texts:
                local_rng = np.random.RandomState(abs(hash(t)) % (2**32))
                vec = local_rng.rand(dim)
                if "movie" in t.lower() or "film" in t.lower():
                    vec[:4] += 5.0
                out.append(vec)
            return np.array(out)

    return FakeEmbedder()


def test_embedding_lr_classifier_roundtrip(tmp_path):
    clf = EmbeddingLRClassifier(model_name="fake")
    clf._embedder = make_fake_embedder()  # bypass real (network) loading

    texts = [
        "what movie won best picture in 2020",
        "recommend a good film tonight",
        "what's the weather in paris",
        "how do i boil an egg",
        "best film about space travel",
        "is it going to rain tomorrow",
    ]
    labels = [
        Label.RELEVANT,
        Label.RELEVANT,
        Label.OUT_OF_DOMAIN,
        Label.OUT_OF_DOMAIN,
        Label.RELEVANT,
        Label.OUT_OF_DOMAIN,
    ]

    try:
        clf.predict_proba(["test"])
        raise AssertionError("Expected ModelNotFittedError")
    except ModelNotFittedError:
        pass
    print("OK: predict before fit raises ModelNotFittedError")

    clf.fit(texts, labels)
    proba = clf.predict_proba(["a new movie about robots"])
    assert 0.0 <= proba[0] <= 1.0
    print(f"OK: fit + predict_proba works, sample P(relevant)={proba[0]:.3f}")

    save_path = tmp_path / "lr_artifact"
    clf.save(save_path)
    loaded = EmbeddingLRClassifier.load(save_path)
    loaded._embedder = clf._embedder  # bypass real loading again
    proba2 = loaded.predict_proba(["a new movie about robots"])
    assert abs(proba[0] - proba2[0]) < 1e-9
    print("OK: save/load roundtrip preserves predictions")

    try:
        clf.fit(["a"], [Label.RELEVANT, Label.OUT_OF_DOMAIN])
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: fit rejects mismatched texts/labels length")

    try:
        clf.fit(["only one class"] * 3, [Label.RELEVANT] * 3)
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: fit rejects single-class training data")

    return clf


def test_tier1_ensemble(fitted_clf):
    config = Tier1Config(relevant_threshold=0.6, out_of_domain_threshold=0.4)
    ensemble = Tier1Ensemble(lr_classifier=fitted_clf, config=config)
    result = ensemble.score("any new movie recommendations?")
    assert result.label in (Label.RELEVANT, Label.OUT_OF_DOMAIN, None)
    assert result.ft_prob is None and result.disagreement is None
    print(
        f"OK: Tier1Ensemble.score (LR-only) -> "
        f"ensemble_score={result.ensemble_score:.3f}, label={result.label}"
    )

    try:
        Tier1Ensemble(lr_classifier=EmbeddingLRClassifier(), config=config)
        raise AssertionError("Expected ModelNotFittedError")
    except ModelNotFittedError:
        pass
    print("OK: Tier1Ensemble rejects an unfitted lr_classifier")

    return ensemble


def test_full_orchestrator_with_fake_tier2(ensemble):
    config = RelevanceClassifierConfig(
        tier1=ensemble.config,
        tier2=Tier2Config(relevant_threshold=0.6, out_of_domain_threshold=0.4),
    )

    class FakeTier2Always:
        def __init__(self, value):
            self.value = value

        def score(self, text):
            return self.value

    # Force the "Tier1 uncertain -> Tier2 also uncertain -> default policy" path.
    clf = QueryRelevanceClassifier(
        tier1=ensemble, tier2=FakeTier2Always(0.5), config=config
    )
    result = clf.classify("some ambiguous query about a celebrity")
    assert result.label in (Label.RELEVANT, Label.OUT_OF_DOMAIN)
    print(
        f"OK: full orchestrator -> label={result.label}, resolved_by={result.resolved_by}"
    )

    try:
        clf.classify("")
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: orchestrator rejects empty query")

    # Exploration sampling: with exploration_rate=1.0, every OUT_OF_DOMAIN
    # decision should be flagged as an exploration sample.
    import random

    explore_config = RelevanceClassifierConfig(
        tier1=ensemble.config,
        tier2=Tier2Config(relevant_threshold=0.6, out_of_domain_threshold=0.4),
        default_policy_label="out_of_domain",
    )
    explore_config.feedback.exploration_rate = 1.0
    clf_explore = QueryRelevanceClassifier(
        tier1=ensemble,
        tier2=FakeTier2Always(0.5),
        config=explore_config,
        rng=random.Random(0),
    )
    result2 = clf_explore.classify("some totally unrelated ambiguous query")
    if result2.label == Label.OUT_OF_DOMAIN:
        assert result2.is_exploration_sample is True
        assert clf_explore.should_call_llm(result2) is True
    print(
        f"OK: exploration sampling -> label={result2.label}, "
        f"is_exploration_sample={result2.is_exploration_sample}"
    )


def test_data_generation_parsing():
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return '```json\n["example one", "example two", "example one"]\n```'

    examples = generate_synthetic_dataset(
        llm_call_fn=fake_llm,
        positive_categories={
            "movie_trivia": DEFAULT_POSITIVE_CATEGORIES["movie_trivia"]
        },
        negative_categories={"cooking": "questions about cooking"},
        n_per_category=3,
    )
    # Same fake response across categories -> global dedup means only the
    # first category's 2 unique strings survive.
    assert len(examples) == 2
    print(
        f"OK: data generation produced {len(examples)} deduped examples "
        f"after {calls['n']} LLM calls"
    )

    def flaky_then_ok(prompt):
        flaky_then_ok.calls += 1
        if flaky_then_ok.calls == 1:
            raise RuntimeError("simulated transient LLM failure")
        return '["query a", "query b"]'

    flaky_then_ok.calls = 0
    examples2 = generate_synthetic_dataset(
        llm_call_fn=flaky_then_ok,
        positive_categories={"movie_trivia": "movie trivia"},
        negative_categories={},  # deliberately empty -- must NOT fall back to defaults
        n_per_category=2,
        max_retries=2,
    )
    assert len(examples2) == 2
    assert flaky_then_ok.calls == 2, (
        f"expected exactly 2 LLM calls (1 retry, 1 success, no fallback to "
        f"default negative categories), got {flaky_then_ok.calls}"
    )
    print("OK: data generation retries past a transient LLM failure")
    print("OK: explicitly empty category dict is NOT replaced by defaults")

    def always_malformed(prompt):
        return "not json at all"

    def all_categories_fail():
        from relevance_gate.exceptions import DataGenerationError

        try:
            generate_synthetic_dataset(
                llm_call_fn=always_malformed,
                positive_categories={"x": "x"},
                negative_categories={"y": "y"},
                n_per_category=2,
                max_retries=0,
            )
            raise AssertionError("Expected DataGenerationError")
        except DataGenerationError:
            pass

    all_categories_fail()
    print("OK: data generation raises when every category fails")


if __name__ == "__main__":
    test_config_validation()
    fitted = test_embedding_lr_classifier_roundtrip(pathlib.Path(tempfile.mkdtemp()))
    ens = test_tier1_ensemble(fitted)
    test_full_orchestrator_with_fake_tier2(ens)
    test_data_generation_parsing()
    print("\nAll smoke tests passed.")
