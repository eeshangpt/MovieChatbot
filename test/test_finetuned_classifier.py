"""Smoke test for FineTunedClassifier's training loop, inference, and
save/load roundtrip -- using a tiny, randomly-initialized DistilBERT
(constructed locally, no Hugging Face Hub download needed) and a minimal
hash-based fake tokenizer standing in for a real pretrained one.

This validates the PyTorch mechanics (weighted loss, optimizer step,
batching, save/load) structurally. It does NOT validate real-world
classification accuracy -- that requires the actual pretrained weights,
which need network access to huggingface.co (test on your own machine /
CI with internet access for that).
"""

import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

import relevance_gate.finetuned_classifier as ftc_module
from relevance_gate.exceptions import InvalidInputError, ModelNotFittedError
from relevance_gate.finetuned_classifier import FineTunedClassifier
from relevance_gate.schemas import Label

_VOCAB_SIZE = 500
_FAKE_MAX_LEN = 32


class _FakeTokenizer:
    """Minimal stand-in: hashes whitespace tokens into a tiny vocab range.

    Not linguistically meaningful, but exercises the same tensor shapes
    and call signature (padding/truncation/return_tensors) the real
    tokenizer would produce.
    """

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=_FAKE_MAX_LEN,
        return_tensors="pt",
    ):
        all_ids = []
        for t in texts:
            ids = [
                abs(hash(tok)) % (_VOCAB_SIZE - 10) + 10 for tok in t.lower().split()
            ]
            ids = [1] + ids[: max_length - 2] + [2]  # fake CLS/SEP
            all_ids.append(ids)

        max_len_in_batch = max(len(ids) for ids in all_ids)
        input_ids, attention_mask = [], []
        for ids in all_ids:
            pad_len = max_len_in_batch - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def save_pretrained(self, path):
        # Real tokenizers persist vocab files; nothing to persist for this
        # fake one, but the call must exist so FineTunedClassifier.save()
        # doesn't blow up.
        pass


def _make_tiny_model_and_tokenizer():
    config = DistilBertConfig(
        vocab_size=_VOCAB_SIZE,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = DistilBertForSequenceClassification(config)
    return model, _FakeTokenizer()


def test_input_validation():
    clf = FineTunedClassifier()
    try:
        clf.predict_proba(["test"])
        raise AssertionError("Expected ModelNotFittedError")
    except ModelNotFittedError:
        pass
    print("OK: predict before fit raises ModelNotFittedError")

    try:
        clf.fit(["a", "b"], [Label.RELEVANT])
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: fit rejects mismatched texts/labels length")

    try:
        clf.fit(["a", "b", "c"], [Label.RELEVANT] * 3)
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: fit rejects single-class training data")

    try:
        clf.fit(["a", "b"], [Label.RELEVANT, Label.OUT_OF_DOMAIN], epochs=0)
        raise AssertionError("Expected InvalidInputError")
    except InvalidInputError:
        pass
    print("OK: fit rejects epochs=0")


def test_training_inference_and_roundtrip(tmp_path):
    texts = [
        "what movie won best picture",
        "recommend a good film",
        "best film about space",
        "who directed that movie",
        "what's the weather today",
        "how do i boil an egg",
        "is it raining tomorrow",
        "help with my math homework",
    ]
    labels = [
        Label.RELEVANT,
        Label.RELEVANT,
        Label.RELEVANT,
        Label.RELEVANT,
        Label.OUT_OF_DOMAIN,
        Label.OUT_OF_DOMAIN,
        Label.OUT_OF_DOMAIN,
        Label.OUT_OF_DOMAIN,
    ]
    weights = [1.0, 1.0, 0.9, 0.6, 1.0, 1.0, 0.9, 0.6]

    clf = FineTunedClassifier(model_name="fake-tiny-distilbert", device="cpu")
    model, tokenizer = _make_tiny_model_and_tokenizer()
    clf._model, clf._tokenizer = model, tokenizer  # bypass real (network) loading

    clf.fit(
        texts,
        labels,
        sample_weight=weights,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        validation_split=0.25,
    )
    assert clf.is_fitted
    print("OK: training loop completed without error (weighted loss, 2 epochs)")

    probs = clf.predict_proba(["a new movie about robots", "what time is it"])
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    print(f"OK: predict_proba returns valid probabilities: {probs}")

    # Single-example convenience method.
    single = clf.predict_proba_single("any good movies out right now")
    assert 0.0 <= single <= 1.0
    print(f"OK: predict_proba_single works: {single:.3f}")

    save_path = tmp_path / "ft_artifact"
    clf.save(save_path)
    assert (save_path / "metadata.json").exists()
    assert (save_path / "config.json").exists()  # written by HF save_pretrained
    print("OK: save() wrote model + metadata to disk")

    loaded = FineTunedClassifier.load(save_path, device="cpu")
    assert loaded.is_fitted
    # The fake tokenizer's save_pretrained() is a no-op (it has no real vocab
    # to persist), so AutoTokenizer.from_pretrained() on load falls back to a
    # different tokenizer than the one training actually used -- that's a
    # limitation of this mock, not of FineTunedClassifier. Swap the *real*
    # fake tokenizer back in to isolate the check to the model weights
    # (which DO go through a real HF save_pretrained/from_pretrained
    # roundtrip and should match exactly).
    loaded._tokenizer = tokenizer
    loaded_probs = loaded.predict_proba(["a new movie about robots", "what time is it"])
    assert all(abs(a - b) < 1e-5 for a, b in zip(probs, loaded_probs))
    print(
        "OK: save/load roundtrip preserves model weights exactly (real HF serialization)"
    )


def test_oom_backoff_path_is_reachable():
    """Verify the OOM-handling branch's logic is sound by simulating a
    RuntimeError('CUDA out of memory') on the first forward call.
    """
    texts = ["movie question", "weather question", "film question", "math question"]
    labels = [Label.RELEVANT, Label.OUT_OF_DOMAIN, Label.RELEVANT, Label.OUT_OF_DOMAIN]

    clf = FineTunedClassifier(model_name="fake-tiny-distilbert", device="cpu")
    model, tokenizer = _make_tiny_model_and_tokenizer()
    clf._model, clf._tokenizer = model, tokenizer

    call_count = {"n": 0}
    original_forward = model.forward

    def flaky_forward(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return original_forward(*args, **kwargs)

    model.forward = flaky_forward
    try:
        clf.fit(texts, labels, epochs=1, batch_size=4, validation_split=0.0)
        assert clf.is_fitted
        print("OK: training recovers from a simulated CUDA OOM via batch-size backoff")
    finally:
        model.forward = original_forward


if __name__ == "__main__":
    test_input_validation()
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        test_training_inference_and_roundtrip(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    test_oom_backoff_path_is_reachable()
    print("\nAll FineTunedClassifier smoke tests passed.")
