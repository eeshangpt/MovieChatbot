"""
Shared data contracts used across the query relevance classification system.
Author: Eeshan Gupta
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .exceptions import InvalidInputError


class Label(Enum):
    """Final, actionable classification label for a user query."""

    RELEVANT = "relevant"
    OUT_OF_DOMAIN = "out_of_domain"


class TierStage(Enum):
    """Which stage of the cascade produced the final decision."""

    TIER1 = "tier1"
    TIER2 = "tier2"
    DEFAULT_POLICY = "default_policy"


class LabelSource(Enum):
    """Provenance of a training label, used to weight examples during training."""

    SYNTHETIC = "synthetic"
    PRODUCTION_HEURISTIC = "production_heuristic"
    PRODUCTION_LLM_JUDGE = "production_llm_judge"
    MANUAL = "manual"


# Default sample weights applied to training examples by source. Production
# labels derived from a single heuristic pattern match are noisier than
# curated synthetic data or an LLM-judge call, so they're downweighted.
DEFAULT_SOURCE_WEIGHTS = {
    LabelSource.SYNTHETIC: 1.0,
    LabelSource.PRODUCTION_LLM_JUDGE: 0.9,
    LabelSource.PRODUCTION_HEURISTIC: 0.6,
    LabelSource.MANUAL: 1.0,
}


@dataclass
class ClassificationResult:
    """Result of running a query through the full relevance classification cascade."""

    query: str
    label: Label
    confidence: float
    resolved_by: TierStage
    tier1_lr_prob: Optional[float] = None
    tier1_ft_prob: Optional[float] = None
    tier1_ensemble_score: Optional[float] = None
    tier1_disagreement: Optional[float] = None
    tier2_score: Optional[float] = None
    is_exploration_sample: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise InvalidInputError(
                "ClassificationResult.query must be a non-empty string."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidInputError(
                f"ClassificationResult.confidence must be in [0, 1], got {self.confidence}."
            )

    def to_dict(self) -> dict:
        """Serialize to a flat, JSON-friendly dict (e.g. for logging)."""
        return {
            "query": self.query,
            "label": self.label.value,
            "confidence": self.confidence,
            "resolved_by": self.resolved_by.value,
            "tier1_lr_prob": self.tier1_lr_prob,
            "tier1_ft_prob": self.tier1_ft_prob,
            "tier1_ensemble_score": self.tier1_ensemble_score,
            "tier1_disagreement": self.tier1_disagreement,
            "tier2_score": self.tier2_score,
            "is_exploration_sample": self.is_exploration_sample,
            "timestamp": self.timestamp,
        }


@dataclass
class TrainingExample:
    """A single labeled example used to train/retrain Tier 1 classifiers."""

    text: str
    label: Label
    source: LabelSource
    weight: Optional[float] = None
    category: Optional[str] = None  # e.g. "movie_trivia", "hard_negative_celebrity"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise InvalidInputError("TrainingExample.text must be a non-empty string.")
        if self.weight is None:
            self.weight = DEFAULT_SOURCE_WEIGHTS[self.source]
        if not (0.0 < self.weight <= 1.0):
            raise InvalidInputError(
                f"TrainingExample.weight must be in (0, 1], got {self.weight}."
            )
