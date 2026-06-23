"""Top-level orchestrator: runs the full Tier1 -> Tier2 -> default-policy
cascade and produces a final ClassificationResult.

This is the main entry point meant to be called before invoking the
expensive downstream LLM:

    result = classifier.classify(query)
    if classifier.should_call_llm(result):
        response = call_expensive_llm(query)
        ...
"""

from __future__ import annotations

import random
from typing import Optional

from .config import RelevanceClassifierConfig
from .exceptions import InvalidInputError
from .schemas import ClassificationResult, Label, TierStage
from .tier1_ensemble import Tier1Ensemble
from .utils import get_logger
from .zero_shot_classifier import ZeroShotRelevanceChecker

logger = get_logger(__name__)


class QueryRelevanceClassifier:
    """Full cascade: Tier1 ensemble -> Tier2 zero-shot -> default policy."""

    def __init__(
        self,
        tier1: Tier1Ensemble,
        tier2: ZeroShotRelevanceChecker,
        config: RelevanceClassifierConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        if tier1 is None or tier2 is None:
            raise InvalidInputError(
                "QueryRelevanceClassifier requires non-None tier1 and tier2 instances."
            )
        self.tier1 = tier1
        self.tier2 = tier2
        self.config = config
        self._rng = rng or random.Random()

    def _resolve_tier2_label(self, score: float) -> Optional[Label]:
        cfg = self.config.tier2
        if score >= cfg.relevant_threshold:
            return Label.RELEVANT
        if score <= cfg.out_of_domain_threshold:
            return Label.OUT_OF_DOMAIN
        return None

    def _default_label(self) -> Label:
        return (
            Label.RELEVANT
            if self.config.default_policy_label == "relevant"
            else Label.OUT_OF_DOMAIN
        )

    def classify(self, query: str) -> ClassificationResult:
        if not isinstance(query, str) or not query.strip():
            raise InvalidInputError("Query must be a non-empty string.")

        tier1_result = self.tier1.score(query)

        if tier1_result.label is not None:
            label = tier1_result.label
            confidence = (
                tier1_result.ensemble_score
                if label == Label.RELEVANT
                else 1.0 - tier1_result.ensemble_score
            )
            result = ClassificationResult(
                query=query,
                label=label,
                confidence=confidence,
                resolved_by=TierStage.TIER1,
                tier1_lr_prob=tier1_result.lr_prob,
                tier1_ft_prob=tier1_result.ft_prob,
                tier1_ensemble_score=tier1_result.ensemble_score,
                tier1_disagreement=tier1_result.disagreement,
            )
        else:
            logger.debug("Tier1 uncertain for query, escalating to Tier2.")
            tier2_score = self.tier2.score(query)
            tier2_label = self._resolve_tier2_label(tier2_score)

            if tier2_label is not None:
                label = tier2_label
                confidence = (
                    tier2_score if label == Label.RELEVANT else 1.0 - tier2_score
                )
                resolved_by = TierStage.TIER2
            else:
                logger.debug("Tier2 also uncertain, applying default policy.")
                label = self._default_label()
                confidence = 0.5
                resolved_by = TierStage.DEFAULT_POLICY

            result = ClassificationResult(
                query=query,
                label=label,
                confidence=confidence,
                resolved_by=resolved_by,
                tier1_lr_prob=tier1_result.lr_prob,
                tier1_ft_prob=tier1_result.ft_prob,
                tier1_ensemble_score=tier1_result.ensemble_score,
                tier1_disagreement=tier1_result.disagreement,
                tier2_score=tier2_score,
            )

        # Exploration sampling: occasionally let an OUT_OF_DOMAIN decision
        # through anyway, purely to collect ground-truth labels downstream
        # and detect drift in queries we'd otherwise never see again.
        if (
            result.label == Label.OUT_OF_DOMAIN
            and self._rng.random() < self.config.feedback.exploration_rate
        ):
            result.is_exploration_sample = True
            logger.debug("Query selected as an exploration sample.")

        return result

    def should_call_llm(self, result: ClassificationResult) -> bool:
        """Convenience helper: should this query be routed to the LLM?"""
        return result.label == Label.RELEVANT or result.is_exploration_sample
