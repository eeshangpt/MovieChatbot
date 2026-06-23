"""Tier 1a: embedding-based classifier (sentence-transformer + logistic regression).

This is the cheapest, fastest primary signal in the cascade -- it embeds the
query with a small sentence-transformer and scores it with a logistic
regression head trained on labeled examples.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import (
    EmbeddingError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFittedError,
    PersistenceError,
)
from .schemas import Label
from .utils import get_logger, resolve_device

logger = get_logger(__name__)

_JOBLIB_FILENAME = "lr_head.joblib"
_METADATA_FILENAME = "metadata.json"


class EmbeddingLRClassifier:
    """Sentence-embedding + logistic regression binary relevance classifier."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
        lr_C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.lr_C = lr_C
        self.random_state = random_state

        self._embedder = None  # lazy-loaded
        self._classifier = None  # lazy-fitted sklearn LogisticRegression
        self._classes: Optional[List[str]] = None
        self._is_fitted = False

    # -- model loading -------------------------------------------------------

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "sentence-transformers is not installed. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        try:
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001 - re-raised as a specific type below
            raise ModelLoadError(
                f"Failed to load embedder '{self.model_name}' on device "
                f"'{self.device}': {exc}"
            ) from exc
        return self._embedder

    # -- embedding -------------------------------------------------------------

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise InvalidInputError("Cannot embed an empty list of texts.")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise InvalidInputError("All texts must be non-empty strings.")

        embedder = self._load_embedder()
        batch_size = self.batch_size

        while True:
            try:
                embeddings = embedder.encode(
                    list(texts),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return embeddings
            except RuntimeError as exc:
                message = str(exc).lower()
                is_oom = "out of memory" in message
                if not is_oom or batch_size <= 1:
                    raise EmbeddingError(
                        f"Embedding failed for {len(texts)} texts: {exc}"
                    ) from exc
                # Back off and retry with a smaller batch before giving up.
                logger.warning(
                    "CUDA OOM while embedding with batch_size=%d; retrying with %d.",
                    batch_size,
                    max(1, batch_size // 2),
                )
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                batch_size = max(1, batch_size // 2)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"Unexpected error while embedding {len(texts)} texts: {exc}"
                ) from exc

    # -- training ---------------------------------------------------------

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[Label],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "EmbeddingLRClassifier":
        if len(texts) != len(labels):
            raise InvalidInputError(
                f"texts and labels must be the same length, got {len(texts)} "
                f"and {len(labels)}."
            )
        if sample_weight is not None and len(sample_weight) != len(texts):
            raise InvalidInputError(
                f"sample_weight must match texts length, got {len(sample_weight)} "
                f"vs {len(texts)}."
            )
        if len(texts) == 0:
            raise InvalidInputError("Cannot fit on an empty dataset.")

        label_values = [label.value for label in labels]
        unique_labels = set(label_values)
        if unique_labels != {Label.RELEVANT.value, Label.OUT_OF_DOMAIN.value}:
            raise InvalidInputError(
                "Training data must contain both RELEVANT and OUT_OF_DOMAIN "
                f"examples; found classes: {unique_labels}."
            )

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ModelLoadError(
                "scikit-learn is not installed. Install it with `pip install scikit-learn`."
            ) from exc

        embeddings = self._embed(texts)

        try:
            classifier = LogisticRegression(
                C=self.lr_C,
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
            classifier.fit(embeddings, label_values, sample_weight=sample_weight)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInputError(
                f"Failed to fit logistic regression head: {exc}"
            ) from exc

        self._classifier = classifier
        self._classes = list(classifier.classes_)
        self._is_fitted = True
        logger.info(
            "Fitted EmbeddingLRClassifier on %d examples (classes=%s).",
            len(texts),
            self._classes,
        )
        return self

    # -- inference --------------------------------------------------------

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(RELEVANT) for each input text, shape (n_texts,)."""
        if not self._is_fitted or self._classifier is None:
            raise ModelNotFittedError(
                "EmbeddingLRClassifier has not been fitted or loaded. "
                "Call .fit(...) or .load(...) first."
            )

        embeddings = self._embed(texts)
        try:
            proba = self._classifier.predict_proba(embeddings)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInputError(f"Prediction failed: {exc}") from exc

        relevant_idx = self._classes.index(Label.RELEVANT.value)
        return proba[:, relevant_idx]

    def predict_proba_single(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted classifier.")
        try:
            import joblib
        except ImportError as exc:
            raise PersistenceError(
                "joblib is not installed. Install it with `pip install joblib`."
            ) from exc

        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._classifier, path / _JOBLIB_FILENAME)
            metadata = {
                "model_name": self.model_name,
                "classes": self._classes,
                "lr_C": self.lr_C,
                "random_state": self.random_state,
                "saved_at": time.time(),
            }
            with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to save classifier to {path}: {exc}"
            ) from exc

        logger.info("Saved EmbeddingLRClassifier to %s.", path)

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "EmbeddingLRClassifier":
        try:
            import joblib
        except ImportError as exc:
            raise PersistenceError(
                "joblib is not installed. Install it with `pip install joblib`."
            ) from exc

        path = Path(path)
        metadata_path = path / _METADATA_FILENAME
        classifier_path = path / _JOBLIB_FILENAME

        if not metadata_path.exists() or not classifier_path.exists():
            raise PersistenceError(
                f"No saved EmbeddingLRClassifier found at {path} "
                f"(expected {_METADATA_FILENAME} and {_JOBLIB_FILENAME})."
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            classifier = joblib.load(classifier_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Failed to load classifier from {path}: {exc}"
            ) from exc

        instance = cls(
            model_name=metadata["model_name"],
            device=device,
            lr_C=metadata.get("lr_C", 1.0),
            random_state=metadata.get("random_state", 42),
        )
        instance._classifier = classifier
        instance._classes = metadata["classes"]
        instance._is_fitted = True
        logger.info("Loaded EmbeddingLRClassifier from %s.", path)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
