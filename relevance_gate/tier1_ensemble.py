"""Tier 1: weighted ensemble combining the embedding+LR classifier with an
optional fine-tuned transformer classifier.

Until a fine-tuned model exists (w_ft=0 / finetuned_classifier=None), this
reduces to just the embedding+LR score. Once a fine-tuned model is trained
and validated against held-out production data, plug it in here and raise
Tier1Config.w_ft accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

from .config import Tier1Config
from .embedding_classifier import EmbeddingLRClassifier
from .exceptions import InvalidInputError, ModelNotFittedError
from .schemas import Label
from .utils import get_logger

if TYPE_CHECKING:
    from .finetuned_classifier import FineTunedClassifier

logger = get_logger(__name__)


@dataclass
class Tier1Result:
    """Per-query output of the Tier 1 ensemble."""

    lr_prob: float
    ft_prob: Optional[float]
    ensemble_score: float
    disagreement: Optional[float]
    label: Optional[Label]  # None means "uncertain" -- escalate to Tier 2


class Tier1Ensemble:
    """Combines Tier 1a (embedding+LR) and Tier 1b (fine-tuned) scores."""

    def __init__(
        self,
        lr_classifier: EmbeddingLRClassifier,
        config: Tier1Config,
        finetuned_classifier: Optional["FineTunedClassifier"] = None,
    ) -> None:
        if lr_classifier is None:
            raise InvalidInputError("Tier1Ensemble requires a non-None lr_classifier.")
        if not lr_classifier.is_fitted:
            raise ModelNotFittedError(
                "Tier1Ensemble's lr_classifier must be fitted/loaded before use."
            )
        self.lr_classifier = lr_classifier
        self.finetuned_classifier = finetuned_classifier
        self.config = config

        if finetuned_classifier is not None and not finetuned_classifier.is_fitted:
            raise ModelNotFittedError(
                "Tier1Ensemble's finetuned_classifier must be fitted/loaded before use."
            )
        if finetuned_classifier is None and config.w_ft > 0:
            logger.warning(
                "Tier1Config.w_ft=%.2f but no finetuned_classifier was provided; "
                "falling back to LR-only scoring.",
                config.w_ft,
            )

    def _classify_band(self, score: float) -> Optional[Label]:
        if score >= self.config.relevant_threshold:
            return Label.RELEVANT
        if score <= self.config.out_of_domain_threshold:
            return Label.OUT_OF_DOMAIN
        return None

    def score(self, text: str) -> Tier1Result:
        if not text or not text.strip():
            raise InvalidInputError("Tier1Ensemble.score requires a non-empty string.")

        lr_prob = self.lr_classifier.predict_proba_single(text)

        ft_prob: Optional[float] = None
        disagreement: Optional[float] = None
        if self.finetuned_classifier is not None:
            ft_prob = self.finetuned_classifier.predict_proba_single(text)
            disagreement = abs(lr_prob - ft_prob)
            w_lr, w_ft = self.config.w_lr, self.config.w_ft
            total = w_lr + w_ft
            ensemble_score = (w_lr * lr_prob + w_ft * ft_prob) / total
        else:
            ensemble_score = lr_prob

        label = self._classify_band(ensemble_score)

        # Even if the blended score looks confident, strong disagreement
        # between the two models is itself an uncertainty signal -- escalate
        # to Tier 2 rather than trust an ensemble average that's masking
        # real disagreement (a cheap form of query-by-committee).
        if (
            disagreement is not None
            and disagreement >= self.config.disagreement_threshold
        ):
            logger.debug(
                "Tier1 disagreement %.3f >= threshold %.3f; escalating to Tier2.",
                disagreement,
                self.config.disagreement_threshold,
            )
            label = None

        return Tier1Result(
            lr_prob=lr_prob,
            ft_prob=ft_prob,
            ensemble_score=ensemble_score,
            disagreement=disagreement,
            label=label,
        )

    def score_batch(self, texts: Sequence[str]) -> List[Tier1Result]:
        return [self.score(t) for t in texts]
