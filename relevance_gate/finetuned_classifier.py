"""Tier 1b: fine-tuned small transformer classifier (e.g. DistilBERT/ELECTRA-small).

This is the second model in the Tier 1 ensemble. It starts at w_ft=0 (see
Tier1Config) until enough real production-labeled data exists to train and
validate it; once validated, raise w_ft to blend it into the ensemble
alongside the embedding+LR classifier.

A manual PyTorch training loop (rather than HF Trainer) is used so that
per-example sample weights -- which downweight noisy production labels
relative to curated synthetic data -- can be folded directly into the loss.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import (
    InferenceError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFittedError,
    PersistenceError,
    TrainingError,
)
from .schemas import Label
from .utils import get_logger, resolve_device

logger = get_logger(__name__)

_METADATA_FILENAME = "metadata.json"

# Fixed, explicit label<->id mapping so save/load and ensemble scoring never
# depend on however sklearn/torch happen to order classes internally.
_LABEL2ID = {Label.OUT_OF_DOMAIN.value: 0, Label.RELEVANT.value: 1}
_ID2LABEL = {v: k for k, v in _LABEL2ID.items()}


class FineTunedClassifier:
    """Fine-tuned transformer binary relevance classifier (Tier 1b)."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "auto",
        max_length: int = 128,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self.random_state = random_state

        self._tokenizer = None
        self._model = None
        self._is_fitted = False

    # -- model loading -------------------------------------------------------

    def _load_base_model(self) -> None:
        """Lazily load the base (not-yet-fine-tuned) tokenizer + model."""
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "transformers is not installed. Install it with `pip install transformers`."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                id2label=_ID2LABEL,
                label2id=_LABEL2ID,
            )
            self._model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load base model '{self.model_name}' on device "
                f"'{self.device}': {exc}"
            ) from exc

    # -- tokenization ------------------------------------------------------

    def _tokenize(self, texts: Sequence[str]):
        try:
            return self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as exc:  # noqa: BLE001
            raise InvalidInputError(f"Tokenization failed: {exc}") from exc

    # -- training -----------------------------------------------------------

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[Label],
        sample_weight: Optional[Sequence[float]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        validation_split: float = 0.1,
        max_retries_on_oom: int = 3,
    ) -> "FineTunedClassifier":
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
        if not 0.0 <= validation_split < 1.0:
            raise InvalidInputError(
                f"validation_split must be in [0, 1), got {validation_split}."
            )
        if epochs < 1 or batch_size < 1:
            raise InvalidInputError("epochs and batch_size must both be >= 1.")

        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
        except ImportError as exc:
            raise ModelLoadError(
                "torch is not installed. Install it with `pip install torch`."
            ) from exc

        self._load_base_model()

        weights = (
            list(sample_weight) if sample_weight is not None else [1.0] * len(texts)
        )
        label_ids = [_LABEL2ID[v] for v in label_values]

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(len(texts))
        n_val = int(len(texts) * validation_split)
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        if len(train_idx) == 0:
            raise InvalidInputError(
                "validation_split leaves zero training examples; reduce it."
            )

        class _QueryDataset(Dataset):
            def __init__(self, idx_subset):
                self.texts = [texts[i] for i in idx_subset]
                self.label_ids = [label_ids[i] for i in idx_subset]
                self.weights = [weights[i] for i in idx_subset]

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, i):
                return self.texts[i], self.label_ids[i], self.weights[i]

        def _collate(batch):
            batch_texts, batch_labels, batch_weights = zip(*batch)
            encodings = self._tokenize(batch_texts)
            return (
                encodings,
                torch.tensor(batch_labels, dtype=torch.long),
                torch.tensor(batch_weights, dtype=torch.float),
            )

        train_ds = _QueryDataset(train_idx)
        val_ds = _QueryDataset(val_idx) if n_val > 0 else None

        try:
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"Failed to construct optimizer: {exc}") from exc

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        current_batch_size = batch_size

        self._model.train()
        for epoch in range(1, epochs + 1):
            loader = DataLoader(
                train_ds,
                batch_size=current_batch_size,
                shuffle=True,
                collate_fn=_collate,
            )
            epoch_loss = 0.0
            n_batches = 0
            oom_retry = False

            for encodings, batch_label_ids, batch_weights in loader:
                try:
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    batch_label_ids = batch_label_ids.to(self.device)
                    batch_weights = batch_weights.to(self.device)

                    optimizer.zero_grad()
                    outputs = self._model(**encodings)
                    per_example_loss = loss_fn(outputs.logits, batch_label_ids)
                    loss = (per_example_loss * batch_weights).mean()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" in message and max_retries_on_oom > 0:
                        logger.warning(
                            "CUDA OOM during training (batch_size=%d); retrying "
                            "epoch with batch_size=%d.",
                            current_batch_size,
                            max(1, current_batch_size // 2),
                        )
                        optimizer.zero_grad()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                        current_batch_size = max(1, current_batch_size // 2)
                        max_retries_on_oom -= 1
                        oom_retry = True
                        break  # restart this epoch's DataLoader with smaller batch
                    raise TrainingError(
                        f"Training failed at epoch {epoch}: {exc}"
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    raise TrainingError(
                        f"Unexpected training failure at epoch {epoch}: {exc}"
                    ) from exc

            if oom_retry:
                continue  # redo this same epoch number with the smaller batch size

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d: avg_loss=%.4f", epoch, epochs, avg_loss)

        self._is_fitted = True

        if val_ds is not None and len(val_ds) > 0:
            val_texts = val_ds.texts
            val_label_values = [_ID2LABEL[i] for i in val_ds.label_ids]
            val_proba = self.predict_proba(val_texts)
            val_preds = [
                Label.RELEVANT.value if p >= 0.5 else Label.OUT_OF_DOMAIN.value
                for p in val_proba
            ]
            val_acc = float(
                np.mean([p == t for p, t in zip(val_preds, val_label_values)])
            )
            logger.info(
                "Validation accuracy after training: %.4f (n=%d)", val_acc, len(val_ds)
            )

        logger.info(
            "Fitted FineTunedClassifier on %d training examples.", len(train_idx)
        )
        return self

    # -- inference ---------------------------------------------------------

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(RELEVANT) for each input text, shape (n_texts,)."""
        if not self._is_fitted:
            raise ModelNotFittedError(
                "FineTunedClassifier has not been fitted or loaded. "
                "Call .fit(...) or .load(...) first."
            )
        if not texts:
            raise InvalidInputError("Cannot predict on an empty list of texts.")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise InvalidInputError("All texts must be non-empty strings.")

        try:
            import torch
        except ImportError as exc:
            raise ModelLoadError(
                "torch is not installed. Install it with `pip install torch`."
            ) from exc

        relevant_id = _LABEL2ID[Label.RELEVANT.value]
        batch_size = 32

        self._model.eval()
        all_probs: List[float] = []
        i = 0
        while i < len(texts):
            batch = list(texts[i : i + batch_size])
            try:
                with torch.no_grad():
                    encodings = self._tokenize(batch)
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    outputs = self._model(**encodings)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, relevant_id]
                    all_probs.extend(probs.cpu().numpy().tolist())
                i += batch_size
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message and batch_size > 1:
                    logger.warning(
                        "CUDA OOM during inference (batch_size=%d); retrying with %d.",
                        batch_size,
                        batch_size // 2,
                    )
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    batch_size = max(1, batch_size // 2)
                    continue
                raise InferenceError(f"Inference failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(f"Unexpected inference failure: {exc}") from exc

        return np.array(all_probs)

    def predict_proba_single(self, text: str) -> float:
        return float(self.predict_proba([text])[0])

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted classifier.")

        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            metadata = {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "random_state": self.random_state,
                "label2id": _LABEL2ID,
                "saved_at": time.time(),
            }
            with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to save classifier to {path}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(
                f"Failed to save classifier to {path}: {exc}"
            ) from exc

        logger.info("Saved FineTunedClassifier to %s.", path)

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "FineTunedClassifier":
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                "transformers is not installed. Install it with `pip install transformers`."
            ) from exc

        path = Path(path)
        metadata_path = path / _METADATA_FILENAME
        if not metadata_path.exists():
            raise PersistenceError(
                f"No saved FineTunedClassifier found at {path} "
                f"(expected {_METADATA_FILENAME})."
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Failed to load metadata from {path}: {exc}"
            ) from exc

        instance = cls(
            model_name=metadata["model_name"],
            device=device,
            max_length=metadata.get("max_length", 128),
            random_state=metadata.get("random_state", 42),
        )

        try:
            instance._tokenizer = AutoTokenizer.from_pretrained(path)
            instance._model = AutoModelForSequenceClassification.from_pretrained(path)
            instance._model.to(instance.device)
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(
                f"Failed to load model weights from {path}: {exc}"
            ) from exc

        instance._is_fitted = True
        logger.info("Loaded FineTunedClassifier from %s.", path)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
