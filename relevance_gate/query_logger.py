"""Persists every classification decision -- and, once available, the
LLM-response-derived label -- as JSON Lines, for use as production training
data in the retraining pipeline.

This is a simple local-file logger appropriate for a single-process
deployment or low/medium query volume. For multi-process or distributed
deployments, swap the file I/O in this class for a proper queue or database
write -- the public interface (`log`, `read_all`, `to_training_examples`)
is intentionally narrow so that's a small change.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional

from .exceptions import InvalidInputError, PersistenceError
from .schemas import ClassificationResult, Label, LabelSource, TrainingExample
from .utils import get_logger

logger = get_logger(__name__)


class QueryLogger:
    """Append-only JSONL logger for query decisions and derived labels."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to create log directory {self.log_path.parent}: {exc}"
            ) from exc

    def log(
        self,
        result: ClassificationResult,
        llm_response: Optional[str] = None,
        derived_label: Optional[Label] = None,
        label_source: Optional[LabelSource] = None,
    ) -> None:
        """Append one record covering the classification decision and,
        if available, the production-derived ground-truth label.

        Args:
            result: the ClassificationResult from QueryRelevanceClassifier.classify().
            llm_response: the downstream LLM's response, if it was called.
            derived_label: the label derived from llm_response via
                label_derivation.derive_label(), if available.
            label_source: provenance of derived_label (required if
                derived_label is set).

        Raises:
            InvalidInputError: if derived_label is set without label_source
                (or vice versa) -- this is almost always a caller bug.
            PersistenceError: if the write to disk fails.
        """
        if (derived_label is None) != (label_source is None):
            raise InvalidInputError(
                "derived_label and label_source must both be set or both be None."
            )

        record = result.to_dict()
        record["llm_response"] = llm_response
        record["derived_label"] = derived_label.value if derived_label else None
        record["label_source"] = label_source.value if label_source else None
        record["logged_at"] = time.time()

        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as exc:
            raise PersistenceError(
                f"Failed to write log entry to {self.log_path}: {exc}"
            ) from exc

    def read_all(
        self,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
    ) -> List[dict]:
        """Read back all logged records (optionally filtered by timestamp).

        Malformed lines are skipped with a warning rather than aborting the
        whole read -- a single corrupted line shouldn't block retraining on
        everything else that's valid.
        """
        if not self.log_path.exists():
            return []

        records: List[dict] = []
        n_skipped = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        n_skipped += 1
                        logger.warning(
                            "Skipping malformed log line %d in %s.",
                            line_num,
                            self.log_path,
                        )
                        continue

                    ts = record.get("timestamp")
                    if min_timestamp is not None and (ts is None or ts < min_timestamp):
                        continue
                    if max_timestamp is not None and (ts is None or ts > max_timestamp):
                        continue
                    records.append(record)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to read log file {self.log_path}: {exc}"
            ) from exc

        if n_skipped:
            logger.warning(
                "Skipped %d malformed log line(s) while reading %s.",
                n_skipped,
                self.log_path,
            )
        return records

    def to_training_examples(
        self,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
    ) -> List[TrainingExample]:
        """Convert logged records with a derived label into TrainingExamples.

        Records without a derived_label (e.g. an OUT_OF_DOMAIN decision that
        wasn't an exploration sample, so no LLM response/label exists) are
        skipped -- they carry no ground truth and would just add noise.
        """
        examples: List[TrainingExample] = []
        n_skipped_no_label = 0

        for record in self.read_all(min_timestamp, max_timestamp):
            derived_label = record.get("derived_label")
            label_source = record.get("label_source")
            query = record.get("query")

            if derived_label is None or label_source is None:
                n_skipped_no_label += 1
                continue
            if not query:
                logger.warning(
                    "Skipping log record with derived label but no query: %s", record
                )
                continue

            try:
                examples.append(
                    TrainingExample(
                        text=query,
                        label=Label(derived_label),
                        source=LabelSource(label_source),
                        metadata={
                            "resolved_by": record.get("resolved_by"),
                            "is_exploration_sample": record.get(
                                "is_exploration_sample"
                            ),
                            "logged_at": record.get("logged_at"),
                        },
                    )
                )
            except (ValueError, InvalidInputError) as exc:
                logger.warning("Skipping malformed log record %s: %s", record, exc)

        if n_skipped_no_label:
            logger.info(
                "Skipped %d log record(s) with no derived label (no ground "
                "truth available).",
                n_skipped_no_label,
            )
        return examples

    def __iter__(self) -> Iterator[dict]:
        return iter(self.read_all())
