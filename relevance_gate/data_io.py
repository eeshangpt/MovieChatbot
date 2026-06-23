"""Serialization helpers for TrainingExample lists (JSON Lines on disk)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .exceptions import InvalidInputError, PersistenceError
from .schemas import Label, LabelSource, TrainingExample
from .utils import get_logger

logger = get_logger(__name__)


def save_training_examples(examples: List[TrainingExample], path: Path) -> None:
    """Save a list of TrainingExample to a JSON Lines file."""
    if not examples:
        raise InvalidInputError("Cannot save an empty list of training examples.")
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "text": ex.text,
                    "label": ex.label.value,
                    "source": ex.source.value,
                    "weight": ex.weight,
                    "category": ex.category,
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise PersistenceError(
            f"Failed to save training examples to {path}: {exc}"
        ) from exc
    logger.info("Saved %d training examples to %s.", len(examples), path)


def load_training_examples(path: Path) -> List[TrainingExample]:
    """Load a list of TrainingExample from a JSON Lines file.

    Malformed lines are skipped with a warning rather than aborting the
    whole load.
    """
    path = Path(path)
    if not path.exists():
        raise PersistenceError(f"Training examples file not found: {path}")

    examples: List[TrainingExample] = []
    n_skipped = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    examples.append(
                        TrainingExample(
                            text=record["text"],
                            label=Label(record["label"]),
                            source=LabelSource(record["source"]),
                            weight=record.get("weight"),
                            category=record.get("category"),
                            metadata=record.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    n_skipped += 1
                    logger.warning(
                        "Skipping malformed line %d in %s: %s", line_num, path, exc
                    )
    except OSError as exc:
        raise PersistenceError(
            f"Failed to read training examples from {path}: {exc}"
        ) from exc

    if n_skipped:
        logger.warning(
            "Skipped %d malformed line(s) while loading %s.", n_skipped, path
        )
    if not examples:
        raise PersistenceError(f"No valid training examples loaded from {path}.")
    return examples
