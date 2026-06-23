"""Tier 2: zero-shot NLI-based relevance check.

Used only for the subset of queries where Tier 1 is uncertain (or Tier 1's
two models disagree), so the extra latency cost is rare, not systemic.
"""

from __future__ import annotations

from typing import List, Sequence

from .config import Tier2Config
from .exceptions import InferenceError, InvalidInputError, ModelLoadError
from .utils import get_logger, resolve_device

logger = get_logger(__name__)


class ZeroShotRelevanceChecker:
    """Wraps a zero-shot NLI pipeline to score 'is this query in-domain?'."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        in_domain_label: str = "movies or TV shows",
        out_domain_label: str = "something other than movies or TV shows",
        hypothesis_template: str = "This query is about {}.",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.in_domain_label = in_domain_label
        self.out_domain_label = out_domain_label
        self.hypothesis_template = hypothesis_template
        self.device = resolve_device(device)
        self._pipeline = None  # lazy-loaded

    @classmethod
    def from_config(
        cls, config: Tier2Config, device: str = "auto"
    ) -> "ZeroShotRelevanceChecker":
        return cls(
            model_name=config.nli_model_name,
            in_domain_label=config.in_domain_label,
            out_domain_label=config.out_domain_label,
            hypothesis_template=config.hypothesis_template,
            device=device,
        )

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ModelLoadError(
                "transformers is not installed. Install it with `pip install transformers`."
            ) from exc

        device_index = 0 if self.device == "cuda" else -1
        try:
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device_index,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load zero-shot model '{self.model_name}' on "
                f"device '{self.device}': {exc}"
            ) from exc
        return self._pipeline

    def score(self, text: str) -> float:
        """Return P(in-domain) for a single query."""
        return self.score_batch([text])[0]

    def score_batch(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            raise InvalidInputError("Cannot score an empty list of texts.")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise InvalidInputError("All texts must be non-empty strings.")

        clf = self._load_pipeline()
        try:
            results = clf(
                list(texts),
                candidate_labels=[self.in_domain_label, self.out_domain_label],
                hypothesis_template=self.hypothesis_template,
                multi_label=False,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                raise InferenceError(
                    "CUDA out of memory during zero-shot classification. "
                    "Consider reducing batch size or moving to CPU."
                ) from exc
            raise InferenceError(f"Zero-shot classification failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Zero-shot classification failed: {exc}") from exc

        if isinstance(results, dict):
            results = [results]

        scores = []
        for result in results:
            try:
                idx = result["labels"].index(self.in_domain_label)
                scores.append(float(result["scores"][idx]))
            except (KeyError, ValueError) as exc:
                raise InferenceError(
                    f"Unexpected zero-shot pipeline output format: {result}"
                ) from exc
        return scores
