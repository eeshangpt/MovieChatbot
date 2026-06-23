"""Evaluation utilities: standard classification metrics plus cascade-specific
operational metrics (how often Tier 2 / the default policy gets invoked --
a key cost/latency signal, since Tier 2 is the expensive fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .exceptions import InvalidInputError
from .schemas import Label


@dataclass
class EvalMetrics:
    """Standard binary classification metrics, RELEVANT treated as positive."""

    n_examples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]], rows=true, cols=pred

    def __str__(self) -> str:
        tn, fp = self.confusion_matrix[0]
        fn, tp = self.confusion_matrix[1]
        return (
            f"n={self.n_examples} | accuracy={self.accuracy:.3f} | "
            f"precision={self.precision:.3f} | recall={self.recall:.3f} | "
            f"f1={self.f1:.3f}\n"
            f"confusion matrix (rows=true, cols=pred) [OUT_OF_DOMAIN, RELEVANT]:\n"
            f"  OUT_OF_DOMAIN: TN={tn:5d}  FP={fp:5d}\n"
            f"  RELEVANT:      FN={fn:5d}  TP={tp:5d}"
        )


@dataclass
class CascadeEvalMetrics(EvalMetrics):
    """EvalMetrics plus operational stats specific to the full 3-tier cascade."""

    tier2_invocation_rate: float = 0.0
    default_policy_rate: float = 0.0
    tier_breakdown: dict = field(default_factory=dict)

    def __str__(self) -> str:
        base = super().__str__()
        return (
            f"{base}\n"
            f"tier2_invocation_rate={self.tier2_invocation_rate:.3f} | "
            f"default_policy_rate={self.default_policy_rate:.3f} | "
            f"tier_breakdown={self.tier_breakdown}"
        )


def _compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise InvalidInputError(
            "scikit-learn is not installed. Install it with `pip install scikit-learn`."
        ) from exc

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def evaluate_probabilistic(
    probs: Sequence[float],
    true_labels: Sequence[Label],
    threshold: float = 0.5,
) -> EvalMetrics:
    """Evaluate a single probabilistic classifier's raw scores against
    ground-truth labels.

    Useful for evaluating Tier1a/Tier1b standalone (e.g. before deciding
    ensemble weights), independent of the full cascade.
    """
    if len(probs) != len(true_labels):
        raise InvalidInputError(
            f"probs and true_labels must be the same length, got {len(probs)} "
            f"and {len(true_labels)}."
        )
    if len(probs) == 0:
        raise InvalidInputError("Cannot evaluate on an empty dataset.")
    if not 0.0 <= threshold <= 1.0:
        raise InvalidInputError(f"threshold must be in [0, 1], got {threshold}.")

    y_true = [1 if label == Label.RELEVANT else 0 for label in true_labels]
    y_pred = [1 if p >= threshold else 0 for p in probs]

    metrics = _compute_metrics(y_true, y_pred)
    return EvalMetrics(n_examples=len(probs), **metrics)


def evaluate_classifier(
    classifier,  # QueryRelevanceClassifier; untyped here to avoid an import cycle
    texts: Sequence[str],
    true_labels: Sequence[Label],
) -> CascadeEvalMetrics:
    """Run the full cascade over a labeled test set and report both
    standard classification metrics and cascade-specific operational
    metrics (how often Tier 2 / the default policy gets invoked).
    """
    if len(texts) != len(true_labels):
        raise InvalidInputError(
            f"texts and true_labels must be the same length, got {len(texts)} "
            f"and {len(true_labels)}."
        )
    if len(texts) == 0:
        raise InvalidInputError("Cannot evaluate on an empty dataset.")

    y_true: List[int] = []
    y_pred: List[int] = []
    tier_counts = {"tier1": 0, "tier2": 0, "default_policy": 0}

    for text, true_label in zip(texts, true_labels):
        result = classifier.classify(text)
        y_true.append(1 if true_label == Label.RELEVANT else 0)
        y_pred.append(1 if result.label == Label.RELEVANT else 0)
        tier_counts[result.resolved_by.value] += 1

    metrics = _compute_metrics(y_true, y_pred)
    n = len(texts)
    return CascadeEvalMetrics(
        n_examples=n,
        tier2_invocation_rate=(tier_counts["tier2"] + tier_counts["default_policy"])
        / n,
        default_policy_rate=tier_counts["default_policy"] / n,
        tier_breakdown=tier_counts,
        **metrics,
    )
