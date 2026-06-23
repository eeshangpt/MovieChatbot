# relevance_gate

A cheap pre-filter that classifies user queries as `RELEVANT` (movie/TV) or
`OUT_OF_DOMAIN` before routing to an expensive downstream LLM, with a
production feedback loop and an MLflow-tracked retraining pipeline.

## Status: complete (all 7 build steps)

1. `config.py` + `schemas.py` -- shared contracts
2. `embedding_classifier.py` -- Tier 1a (sentence-transformer + logistic regression)
3. `data_generation.py` -- LLM-driven synthetic training data
4. `zero_shot_classifier.py` (Tier 2) + `relevance_classifier.py` (orchestrator)
5. `finetuned_classifier.py` -- Tier 1b (fine-tuned DistilBERT/ELECTRA, PyTorch
   training loop with sample-weighted loss), wired into `tier1_ensemble.py`
6. `label_derivation.py` + `query_logger.py` -- production feedback loop
   (heuristic refusal-pattern matching, LLM-judge fallback, JSONL logging)
7. `train_pipeline.py` (MLflow-tracked, manually triggered) + `evaluation.py`
   + `data_io.py` (synthetic data persistence)

## Architecture

See `relevance_gate_architecture.drawio` -- open at https://app.diagrams.net
(File > Open From > Device) for the full Tier1 -> Tier2 -> default-policy
cascade plus the production feedback loop and retraining pipeline.

## Quick start: inference

```python
from relevance_gate import (
    RelevanceClassifierConfig, EmbeddingLRClassifier, FineTunedClassifier,
    Tier1Ensemble, ZeroShotRelevanceChecker, QueryRelevanceClassifier,
    generate_synthetic_dataset, save_training_examples, Label,
)

# 1. Generate (or load) labeled training data.
def my_llm_call(prompt: str) -> str:
    return my_client.generate(prompt)  # wrap whatever LLM client you use

examples = generate_synthetic_dataset(my_llm_call, n_per_category=30)
save_training_examples(examples, "data/synthetic.jsonl")

# 2. Train (see train_pipeline.py for the MLflow-tracked version of this).
texts = [e.text for e in examples]
labels = [e.label for e in examples]
weights = [e.weight for e in examples]

lr_clf = EmbeddingLRClassifier(device="cuda")
lr_clf.fit(texts, labels, sample_weight=weights)
lr_clf.save("artifacts/tier1_lr")

# 3. Wire up the cascade (Tier1b/fine-tuned is optional; omit until trained).
config = RelevanceClassifierConfig()  # tune thresholds as needed; w_ft=0 by default
tier1 = Tier1Ensemble(lr_classifier=lr_clf, config=config.tier1)
tier2 = ZeroShotRelevanceChecker.from_config(config.tier2, device=config.resolved_device)
classifier = QueryRelevanceClassifier(tier1=tier1, tier2=tier2, config=config)

# 4. Classify queries.
result = classifier.classify("who directed Inception?")
if classifier.should_call_llm(result):
    response = call_expensive_llm(result.query)
```

## Quick start: production feedback loop

```python
from relevance_gate import QueryLogger, derive_label

logger = QueryLogger(config.feedback.log_path)

result = classifier.classify(query)
llm_response = None
derived_label, label_source = None, None

if classifier.should_call_llm(result):
    llm_response = call_expensive_llm(query)
    try:
        derived_label, label_source = derive_label(
            query, llm_response, judge_call_fn=my_cheap_judge_llm_call,
        )
    except LabelDerivationError:
        pass  # heuristic was ambiguous and judge failed/wasn't provided -- skip logging a label

logger.log(result, llm_response=llm_response,
           derived_label=derived_label, label_source=label_source)
```

## Quick start: retraining (manual trigger, MLflow-tracked)

```bash
python -m relevance_gate.train_pipeline \
    --synthetic-data-path data/synthetic.jsonl \
    --production-log-path logs/query_log.jsonl \
    --output-dir artifacts/ \
    --train-finetuned \
    --mlflow-tracking-uri sqlite:///mlflow.db
```

This trains Tier1a (and optionally Tier1b), evaluates both standalone and as
a full cascade, logs everything to MLflow, and saves model artifacts to
`--output-dir`. It does **not** auto-promote/register a model version --
inspect the run in the MLflow UI (`mlflow ui --backend-store-uri
sqlite:///mlflow.db`) and promote manually once you're satisfied.

Programmatic equivalent: `relevance_gate.train_pipeline.run_training_pipeline(...)`.

## Error handling

All errors raise specific subclasses of `RelevanceClassifierError`
(`ConfigError`, `ModelLoadError`, `ModelNotFittedError`, `EmbeddingError`,
`InferenceError`, `TrainingError`, `InvalidInputError`, `DataGenerationError`,
`LabelDerivationError`, `PersistenceError`) so callers can catch and handle
each failure mode differently (e.g. retry on `EmbeddingError`/CUDA OOM, fail
fast on `ConfigError`, skip-and-continue on `LabelDerivationError`).

## Testing without network access

These exercise validation/error-handling logic and the core mechanics using
monkeypatched/locally-constructed models -- no Hugging Face Hub or other
network access needed:

- `smoke_test.py` -- config validation, Tier1a (LR) fit/predict/save/load,
  the full Tier1->Tier2->default-policy cascade, exploration sampling, and
  synthetic data generation parsing/retry/dedup logic.
- `test_finetuned_classifier.py` -- Tier1b (fine-tuned) training loop,
  weighted loss, CUDA-OOM batch-size backoff, and save/load roundtrip, using
  a tiny randomly-initialized DistilBERT built locally (no download).
- `test_train_pipeline.py` -- end-to-end `run_training_pipeline()` with a
  monkeypatched embedder, real MLflow run (sqlite backend), and artifact
  persistence.

Run with `python3 <file>.py`. On your own machine (with Hugging Face Hub
access), also do a real run with actual pretrained weights before relying on
this in production -- these tests validate mechanics, not real-world
classification accuracy.

## Notes on the GPU target

- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) and
  `cross-encoder/nli-deberta-v3-xsmall` (~70M params) comfortably coexist
  with plenty of headroom for batching. A fine-tuned DistilBERT/ELECTRA-small
  (~250-300MB) also fits alongside them.
- Both `EmbeddingLRClassifier` and `FineTunedClassifier` automatically back
  off to a smaller batch size on CUDA OOM (training and inference) before
  giving up with a specific exception.

## MLflow note

MLflow 3.x put the plain filesystem tracking backend (`file:./mlruns`) into
maintenance mode and raises by default. `MLflowConfig` defaults to a local
SQLite backend (`sqlite:///mlflow.db`) instead, which needs no extra service.
Swap in a real database URI for a shared/multi-user tracking server.
