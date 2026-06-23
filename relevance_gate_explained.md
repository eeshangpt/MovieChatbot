# The `relevance_gate` Module — Explained for a Fresh Graduate

## The Problem It Solves

This chatbot is built on top of the IMDB dataset — it answers questions about movies and TV shows. But users will inevitably type things like *"what's the weather today?"* or *"help me write a Python script"* — questions it has no business answering.

The naive fix is to just pass every query to your big, powerful LLM and let it figure out what to do. That works, but **LLM calls are expensive and slow**. If you're getting thousands of queries a day, paying to run an LLM on every "what's 2+2?" question is wasteful.

The `relevance_gate` module is a **cheap pre-filter** that classifies each incoming query as either:
- `RELEVANT` — it's about movies/TV, send it to the LLM
- `OUT_OF_DOMAIN` — it's off-topic, reject it immediately without touching the LLM

The goal is to make this decision as fast and cheaply as possible, but still be accurate.

---

## The Big Idea: A Tiered Cascade

Instead of one classifier, there are **three layers**, each only invoked if the previous one wasn't confident enough:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  TIER 1: Fast, cheap classifiers    │  ← Handles ~90%+ of queries
│  (Embedding + Logistic Regression,  │
│   optionally + fine-tuned DistilBERT│
└──────────────┬──────────────────────┘
               │ "I'm not sure"
               ▼
┌─────────────────────────────────────┐
│  TIER 2: Smarter, slower NLI model  │  ← Rare fallback
│  (DeBERTa zero-shot classifier)     │
└──────────────┬──────────────────────┘
               │ "I'm still not sure"
               ▼
┌─────────────────────────────────────┐
│  DEFAULT POLICY: Just pick RELEVANT │  ← Last resort
│  (better to let an odd query through│
│   than to wrongly block a real one) │
└─────────────────────────────────────┘
```

This is called a **cascade**: fast and cheap gates first, escalate to expensive gates only when necessary. The file that orchestrates this whole flow is `relevance_classifier.py` (`QueryRelevanceClassifier`).

---

## Component by Component

### 1. Shared Contracts — `schemas.py` and `config.py`

Before diving into the classifiers, you need to know the vocabulary everyone uses.

**`schemas.py`** defines the data structures passed between components:

- **`Label`** — an enum with two values: `RELEVANT` or `OUT_OF_DOMAIN`. Every component ultimately produces one of these.
- **`ClassificationResult`** — the full output of classifying a single query. It records not just the final label but *which tier decided it*, the confidence score, and all intermediate scores. Think of it as a full audit trail.
- **`TrainingExample`** — a labeled (text, label) pair used to train or retrain classifiers. It also carries a `weight` (some examples are more trustworthy than others) and a `source` (where did this label come from?).
- **`LabelSource`** — where a training label came from: hand-crafted `SYNTHETIC` data, a `PRODUCTION_HEURISTIC` match, an `PRODUCTION_LLM_JUDGE` verdict, or `MANUAL` labeling. Crucially, noisier sources get lower weights (e.g. `PRODUCTION_HEURISTIC = 0.6` vs `SYNTHETIC = 1.0`), so the classifier doesn't trust all training data equally.

**`config.py`** is where all the knobs live. The most important ones:

- **`relevant_threshold` / `out_of_domain_threshold`**: the score bands. If Tier 1's score for a query is ≥ 0.65, it's RELEVANT; ≤ 0.35 it's OUT_OF_DOMAIN; anything in between is "uncertain" → escalate to Tier 2.
- **`disagreement_threshold`**: if Tier 1's two sub-models (LR and fine-tuned) disagree by more than this amount, escalate to Tier 2 even if the blended score looks confident. One model saying 0.9 and the other saying 0.4 is a red flag.
- **`exploration_rate`**: 5% of queries that would be blocked are let through anyway to collect real data (more on this in the feedback loop section).
- **`default_policy_label`**: what to do when even Tier 2 can't decide. Defaults to `"relevant"` — it's better to let a questionable query reach the LLM than to incorrectly block a real movie question.

---

### 2. Tier 1a — `embedding_classifier.py` (`EmbeddingLRClassifier`)

This is the workhorse. Here's how it works step by step:

**Step 1 — Embed the text**: The query (e.g. "who directed Inception?") is passed through `sentence-transformers/all-MiniLM-L6-v2`, a small pre-trained model that converts text into a list of 384 numbers (a vector). Queries with similar meaning end up with similar vectors.

**Step 2 — Score with Logistic Regression**: That vector is fed into a simple Logistic Regression model (from scikit-learn). This is just a linear function with a sigmoid on top — extremely fast. It outputs a probability between 0 and 1 (P(RELEVANT)).

**Why this combination?** The sentence-transformer handles the "understand the meaning" part. The logistic regression handles the "which side of the line is this?" part. Logistic regression is tiny and takes microseconds. The expensive part (the transformer) only has to run once at training time to create the vectors, and at inference time for each new query.

The class handles:
- Lazy loading (models load only when first needed)
- Batched embedding with CUDA OOM backoff (if GPU runs out of memory, it retries with half the batch size)
- `save()` / `load()` for persistence between restarts

---

### 3. Tier 1b — `finetuned_classifier.py` (`FineTunedClassifier`)

This is the *optional* second model in the Tier 1 ensemble. It fine-tunes a small transformer (DistilBERT by default) directly on the labeled training data.

The key difference from Tier 1a: instead of using a pre-trained embedding frozen in place and only training a logistic regression head, this model updates all its weights end-to-end. That makes it potentially more accurate, but much slower to train and larger to host.

**Why is it optional?** When you're starting out, you might not have enough real production data to make fine-tuning worthwhile. The config defaults to `w_ft = 0.0` (weight zero in the ensemble), which means the fine-tuned model's score is completely ignored until you decide to turn it on.

**Per-example sample weights**: Instead of using HuggingFace's `Trainer`, this uses a manual PyTorch training loop. The reason: `Trainer` doesn't support per-example loss weighting out of the box. Here, each training example's loss is multiplied by its weight before being averaged, so a noisy production-heuristic label (weight 0.6) doesn't hurt the model as much as a clean synthetic label (weight 1.0).

---

### 4. Tier 1 Ensemble — `tier1_ensemble.py` (`Tier1Ensemble`)

This class combines Tier 1a and Tier 1b into a single score:

```
ensemble_score = (w_lr * lr_prob + w_ft * ft_prob) / (w_lr + w_ft)
```

If only Tier 1a is active (`w_ft = 0`), this just returns the LR probability directly.

The ensemble also computes **disagreement** = `|lr_prob - ft_prob|`. If this exceeds `disagreement_threshold`, even a confident-looking ensemble score gets overridden and the query escalates to Tier 2. This is a technique called *query-by-committee* — if two models that were trained differently disagree sharply, the query is probably genuinely ambiguous and deserves more scrutiny.

The output is a `Tier1Result`, which is either a definitive label or `None` (uncertain, escalate).

---

### 5. Tier 2 — `zero_shot_classifier.py` (`ZeroShotRelevanceChecker`)

Tier 2 uses a completely different approach: **zero-shot Natural Language Inference (NLI)**.

The model (`cross-encoder/nli-deberta-v3-xsmall`) is not trained on movie/TV data at all. Instead, you give it the query and two candidate labels:
- `"movies or TV shows"`
- `"something other than movies or TV shows"`

And you ask: *"This query is about [candidate label]."* — does this hypothesis follow from the query? The model scores both hypotheses and you take the probability assigned to the in-domain one.

**Why is this the fallback?** It's slower (a cross-encoder scores both the query and the hypothesis together, which is more expensive than embedding separately) but it understands language more flexibly. It doesn't need training data for your specific domain at all.

---

### 6. The Orchestrator — `relevance_classifier.py` (`QueryRelevanceClassifier`)

This ties everything together. The `classify(query)` method:

1. Runs Tier 1. If it gives a confident label, return that.
2. Otherwise, run Tier 2. If it gives a confident label, return that.
3. Otherwise, fall back to the default policy (RELEVANT).
4. Randomly flag ~5% of blocked queries as **exploration samples** (`is_exploration_sample = True`).

The `should_call_llm(result)` helper tells callers whether to forward the query downstream: it's `True` if the label is RELEVANT, or if it's an exploration sample.

---

## The Production Feedback Loop

This is where the system learns from real traffic.

### The Core Problem

You trained the classifier on synthetic data. But real users will phrase things in ways you didn't anticipate. Over time, your classifier drifts and you need to retrain it — but with what data?

Every time a RELEVANT query goes to the LLM, you get something back: the LLM's response. That response is a **signal**. If the LLM answered the question properly, the query was genuinely relevant. If it replied "I can only help with movies and TV shows", the query was probably off-topic and your classifier made a mistake.

### Step 1 — Derive a Label from the LLM Response (`label_derivation.py`)

Two strategies, cheapest first:

1. **Heuristic refusal-pattern matching**: Scan the LLM response for phrases like *"I can only assist with"*, *"that's outside my scope"*, etc. (see `DEFAULT_REFUSAL_PATTERNS`). If found → `OUT_OF_DOMAIN`. If the response is long and doesn't match any refusal pattern → `RELEVANT`.

2. **LLM-as-judge fallback**: If the heuristic is ambiguous (e.g. a suspiciously short response), ask a second, cheaper LLM to judge: *"Did the chatbot engage with this as a movie question, or did it decline?"* It must respond with `{"label": "relevant"}` or `{"label": "out_of_domain"}`.

The label gets tagged with its `LabelSource` so the downstream training pipeline knows how much to trust it.

### Step 2 — Log Everything (`query_logger.py`)

`QueryLogger` appends every classification decision to a JSONL file (one JSON object per line). Each record contains:
- The original query
- The classification result (label, confidence, which tier decided, all scores)
- The LLM's response (if it was called)
- The derived label and its source (if derivation succeeded)

JSONL is a good format here: it's append-only (safe for concurrent writes with the thread lock), human-readable, and easy to stream line-by-line without loading the whole file.

### Step 3 — The Exploration Samples

There's a subtle problem: if you only collect feedback on queries the classifier let through (labeled RELEVANT), you only ever see the classifier's own blind spots when it makes a false-negative mistake. You never see queries it incorrectly blocked (false positives), because those never reach the LLM.

The solution is **exploration sampling**: 5% of OUT_OF_DOMAIN decisions are secretly allowed through anyway. The LLM responds, you derive a label, and if the label comes back RELEVANT, you've discovered a query your classifier was wrongly rejecting. These examples are particularly valuable for retraining.

---

## Training Data — `data_generation.py` and `data_io.py`

Before you have any real traffic, you need training data. `data_generation.py` uses an LLM to generate synthetic examples.

You give it:
- **Positive categories** (in-domain): movie trivia, recommendations, cast/crew questions, release dates, etc.
- **Negative categories** (out-of-domain): cooking, sports, math, coding, *hard negatives* (celebrity net worth questions that mention a famous actor but aren't about their movies, book questions that might have a film adaptation but don't mention it).

The **hard negatives** are critical. A naive classifier might learn "if the query mentions Tom Hanks → RELEVANT". Hard negatives teach it that *"How much is Tom Hanks worth?"* is still out-of-domain even though it names a famous actor.

The LLM is prompted to generate N diverse examples per category as a JSON array. The code handles retries if the LLM returns malformed output, and deduplicates across the whole run.

`data_io.py` handles saving these examples to JSONL and loading them back, so you don't regenerate them on every training run.

---

## The Retraining Pipeline — `train_pipeline.py`

When you decide it's time to retrain (manually — nothing is scheduled automatically), you run:

```bash
python -m relevance_gate.train_pipeline \
    --synthetic-data-path data/synthetic.jsonl \
    --production-log-path logs/query_log.jsonl \
    --output-dir artifacts/
```

What it does:

1. **Load data**: combines synthetic examples and production log examples (only those with a derived label).
2. **Split**: stratified train/test split (80/20 by default), keeping the class ratio the same in both halves.
3. **Train Tier 1a** (EmbeddingLRClassifier) with sample weights.
4. **Optionally train Tier 1b** (FineTunedClassifier) if `--train-finetuned` is passed.
5. **Evaluate**: computes accuracy, precision, recall, F1, confusion matrix. Also evaluates the full cascade (how often does it hit Tier 2? How often does it fall back to the default policy?).
6. **Log everything to MLflow**: every parameter, every metric, every artifact (the saved model files). MLflow gives you a UI to compare runs side-by-side over time.
7. **Save model artifacts to disk** (e.g. `artifacts/tier1_lr/`). Notably, it does **not** auto-promote the model — you inspect the MLflow UI and manually decide whether the new run is better than what's in production.

---

## Error Handling

Every failure mode has its own exception class (all subclasses of `RelevanceClassifierError`):

| Exception | Meaning |
|---|---|
| `ConfigError` | Bad config values (e.g. thresholds out of range) |
| `ModelLoadError` | Missing dependency or model failed to download |
| `ModelNotFittedError` | Tried to run inference before training |
| `EmbeddingError` | Sentence-transformer embedding failed |
| `InferenceError` | Prediction step failed |
| `TrainingError` | Model training failed |
| `InvalidInputError` | Bad inputs (empty query, mismatched lengths, etc.) |
| `DataGenerationError` | LLM failed to produce usable synthetic examples |
| `LabelDerivationError` | Couldn't derive a label from the LLM response |
| `PersistenceError` | Reading/writing files failed |

This granularity matters in production: you might want to retry on `EmbeddingError` (transient GPU OOM), fail fast on `ConfigError` (programming mistake), and silently skip on `LabelDerivationError` (the judge couldn't decide — better to log nothing than to log a wrong label).

---

## Putting It All Together — The Full Lifecycle

```
[1. Generate synthetic data]
        ↓
[2. Train Tier 1a (+ optionally Tier 1b) via train_pipeline.py]
        ↓
[3. Deploy: QueryRelevanceClassifier gates all incoming queries]
        ↓
[4. Every RELEVANT query hits the LLM → get a response]
        ↓
[5. derive_label() extracts a training label from the response]
        ↓
[6. QueryLogger appends (query, label, metadata) to query_log.jsonl]
        ↓
[7. 5% of OUT_OF_DOMAIN queries are exploration samples → also logged]
        ↓
[8. When enough production data accumulates, run train_pipeline.py again]
        ↓
[9. Review in MLflow UI, promote the better model, redeploy]
        ↓
        back to [3]
```

The system gets smarter over time as real queries accumulate, without needing a human to manually label each one.