"""Standalone, manually-triggered, MLflow-tracked training pipeline.

Trains Tier 1a (EmbeddingLRClassifier) and, optionally, Tier 1b
(FineTunedClassifier), evaluates both, and logs params/metrics/artifacts to
MLflow. Designed to be run by hand whenever you decide it's time to
retrain -- nothing here is scheduled automatically.

CLI usage:
    python -m relevance_gate.train_pipeline \\
        --synthetic-data-path data/synthetic.jsonl \\
        --production-log-path logs/query_log.jsonl \\
        --output-dir artifacts/ \\
        --mlflow-tracking-uri file:./mlruns

Programmatic usage:
    from relevance_gate.train_pipeline import run_training_pipeline
    summary = run_training_pipeline(output_dir="artifacts/", synthetic_data_path="data/synthetic.jsonl")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .data_io import load_training_examples
from .embedding_classifier import EmbeddingLRClassifier
from .evaluation import (
    CascadeEvalMetrics,
    EvalMetrics,
    evaluate_classifier,
    evaluate_probabilistic,
)
from .exceptions import (
    InvalidInputError,
    ModelLoadError,
    PersistenceError,
    RelevanceClassifierError,
)
from .query_logger import QueryLogger
from .schemas import TrainingExample
from .utils import get_logger, resolve_device

logger = get_logger(__name__)


def _gather_training_data(
    synthetic_data_path: Optional[Path],
    production_log_path: Optional[Path],
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []

    if synthetic_data_path is not None:
        synthetic = load_training_examples(synthetic_data_path)
        logger.info("Loaded %d synthetic training examples.", len(synthetic))
        examples.extend(synthetic)

    if production_log_path is not None:
        production = QueryLogger(production_log_path).to_training_examples()
        logger.info("Loaded %d production training examples.", len(production))
        examples.extend(production)

    if not examples:
        raise InvalidInputError(
            "No training data available: provide synthetic_data_path and/or "
            "production_log_path with at least one valid example."
        )
    return examples


def _stratified_split(
    examples: List[TrainingExample], test_split: float, random_state: int
):
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ModelLoadError(
            "scikit-learn is not installed. Install it with `pip install scikit-learn`."
        ) from exc

    if not 0.0 < test_split < 1.0:
        raise InvalidInputError(f"test_split must be in (0, 1), got {test_split}.")

    label_values = [e.label.value for e in examples]
    # Stratification requires every class to have at least 2 members.
    counts = {v: label_values.count(v) for v in set(label_values)}
    if any(c < 2 for c in counts.values()):
        raise InvalidInputError(
            f"Each class needs >= 2 examples for a stratified split; got counts={counts}."
        )

    train_examples, test_examples = train_test_split(
        examples,
        test_size=test_split,
        random_state=random_state,
        stratify=label_values,
    )
    return train_examples, test_examples


def _log_mlflow_eval_metrics(mlflow, prefix: str, metrics: EvalMetrics) -> None:
    mlflow.log_metric(f"{prefix}_accuracy", metrics.accuracy)
    mlflow.log_metric(f"{prefix}_precision", metrics.precision)
    mlflow.log_metric(f"{prefix}_recall", metrics.recall)
    mlflow.log_metric(f"{prefix}_f1", metrics.f1)
    mlflow.log_dict(
        {
            "confusion_matrix": metrics.confusion_matrix,
            "n_examples": metrics.n_examples,
        },
        f"{prefix}_confusion_matrix.json",
    )


def run_training_pipeline(
    output_dir: Path,
    synthetic_data_path: Optional[Path] = None,
    production_log_path: Optional[Path] = None,
    test_split: float = 0.2,
    train_finetuned: bool = False,
    embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    finetuned_model_name: str = "distilbert-base-uncased",
    lr_C: float = 1.0,
    finetuned_epochs: int = 3,
    finetuned_batch_size: int = 16,
    finetuned_learning_rate: float = 2e-5,
    device: str = "auto",
    random_state: int = 42,
    mlflow_tracking_uri: str = "file:./mlruns",
    mlflow_experiment_name: str = "query-relevance-classifier",
    evaluate_full_cascade: bool = True,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
) -> dict:
    """Run one full, manually-triggered retraining job tracked in MLflow.

    Returns:
        A summary dict (also logged to MLflow as an artifact) containing the
        run id and key metrics, for convenient programmatic inspection.

    Raises:
        InvalidInputError: bad arguments or insufficient/malformed data.
        ModelLoadError: a required dependency (mlflow, scikit-learn, ...) is
            missing, or a model failed to load.
        PersistenceError: saving trained artifacts to disk failed.
        TrainingError: fine-tuned model training failed (only if
            train_finetuned=True).
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ModelLoadError(
            "mlflow is not installed. Install it with `pip install mlflow`."
        ) from exc

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PersistenceError(
            f"Failed to create output_dir {output_dir}: {exc}"
        ) from exc

    resolved_device = resolve_device(device)

    examples = _gather_training_data(synthetic_data_path, production_log_path)
    n_synthetic = sum(1 for e in examples if e.source.value == "synthetic")
    n_production = len(examples) - n_synthetic

    train_examples, test_examples = _stratified_split(
        examples, test_split, random_state
    )
    train_texts = [e.text for e in train_examples]
    train_labels = [e.label for e in train_examples]
    train_weights = [e.weight for e in train_examples]
    test_texts = [e.text for e in test_examples]
    test_labels = [e.label for e in test_examples]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    summary: dict = {}

    with mlflow.start_run() as run:
        summary["run_id"] = run.info.run_id
        logger.info("Started MLflow run %s.", run.info.run_id)

        mlflow.log_params(
            {
                "n_examples_total": len(examples),
                "n_synthetic": n_synthetic,
                "n_production": n_production,
                "n_train": len(train_examples),
                "n_test": len(test_examples),
                "test_split": test_split,
                "embedder_model_name": embedder_model_name,
                "lr_C": lr_C,
                "random_state": random_state,
                "device": resolved_device,
                "train_finetuned": train_finetuned,
            }
        )

        # -- Tier 1a: EmbeddingLRClassifier ---------------------------------
        lr_clf = EmbeddingLRClassifier(
            model_name=embedder_model_name,
            device=resolved_device,
            lr_C=lr_C,
            random_state=random_state,
        )
        lr_clf.fit(train_texts, train_labels, sample_weight=train_weights)

        lr_test_probs = lr_clf.predict_proba(test_texts)
        lr_metrics = evaluate_probabilistic(lr_test_probs, test_labels)
        logger.info("Tier1a (LR) test metrics:\n%s", lr_metrics)
        _log_mlflow_eval_metrics(mlflow, "tier1a", lr_metrics)
        summary["tier1a_metrics"] = vars(lr_metrics)

        lr_save_path = output_dir / "tier1_lr"
        lr_clf.save(lr_save_path)
        mlflow.log_artifacts(str(lr_save_path), artifact_path="tier1_lr_model")

        # -- Tier 1b: FineTunedClassifier (optional) ------------------------
        ft_clf = None
        if train_finetuned:
            from .finetuned_classifier import FineTunedClassifier

            mlflow.log_params(
                {
                    "finetuned_model_name": finetuned_model_name,
                    "finetuned_epochs": finetuned_epochs,
                    "finetuned_batch_size": finetuned_batch_size,
                    "finetuned_learning_rate": finetuned_learning_rate,
                }
            )
            ft_clf = FineTunedClassifier(
                model_name=finetuned_model_name,
                device=resolved_device,
                random_state=random_state,
            )
            ft_clf.fit(
                train_texts,
                train_labels,
                sample_weight=train_weights,
                epochs=finetuned_epochs,
                batch_size=finetuned_batch_size,
                learning_rate=finetuned_learning_rate,
            )

            ft_test_probs = ft_clf.predict_proba(test_texts)
            ft_metrics = evaluate_probabilistic(ft_test_probs, test_labels)
            logger.info("Tier1b (fine-tuned) test metrics:\n%s", ft_metrics)
            _log_mlflow_eval_metrics(mlflow, "tier1b", ft_metrics)
            summary["tier1b_metrics"] = vars(ft_metrics)

            ft_save_path = output_dir / "tier1_finetuned"
            ft_clf.save(ft_save_path)
            mlflow.log_artifacts(
                str(ft_save_path), artifact_path="tier1_finetuned_model"
            )
        else:
            logger.info("train_finetuned=False; skipping Tier1b.")

        # -- Optional: full cascade evaluation (Tier1 + Tier2 + default) ----
        if evaluate_full_cascade:
            try:
                from .config import RelevanceClassifierConfig, Tier1Config, Tier2Config
                from .relevance_classifier import QueryRelevanceClassifier
                from .tier1_ensemble import Tier1Ensemble
                from .zero_shot_classifier import ZeroShotRelevanceChecker

                tier1_config = Tier1Config(
                    embedder_model_name=embedder_model_name,
                    finetuned_model_name=finetuned_model_name,
                    w_lr=1.0 if ft_clf is None else 0.7,
                    w_ft=0.0 if ft_clf is None else 0.3,
                )
                tier2_config = Tier2Config(nli_model_name=nli_model_name)
                full_config = RelevanceClassifierConfig(
                    tier1=tier1_config, tier2=tier2_config
                )

                tier1_ensemble = Tier1Ensemble(
                    lr_classifier=lr_clf,
                    config=tier1_config,
                    finetuned_classifier=ft_clf,
                )
                tier2 = ZeroShotRelevanceChecker.from_config(
                    tier2_config, device=resolved_device
                )
                cascade = QueryRelevanceClassifier(
                    tier1=tier1_ensemble, tier2=tier2, config=full_config
                )

                cascade_metrics: CascadeEvalMetrics = evaluate_classifier(
                    cascade, test_texts, test_labels
                )
                logger.info("Full cascade test metrics:\n%s", cascade_metrics)
                _log_mlflow_eval_metrics(mlflow, "cascade", cascade_metrics)
                mlflow.log_metric(
                    "cascade_tier2_invocation_rate",
                    cascade_metrics.tier2_invocation_rate,
                )
                mlflow.log_metric(
                    "cascade_default_policy_rate", cascade_metrics.default_policy_rate
                )
                summary["cascade_metrics"] = vars(cascade_metrics)
            except RelevanceClassifierError as exc:
                # Tier 2 (NLI model) may be unavailable in offline/restricted
                # environments -- that shouldn't fail the whole training run,
                # since Tier 1 training/evaluation is this script's main job.
                logger.warning(
                    "Skipping full-cascade evaluation (Tier 2 model unavailable?): %s",
                    exc,
                )
                mlflow.set_tag("cascade_eval_skipped_reason", str(exc))

        mlflow.log_dict(summary, "run_summary.json")
        logger.info("MLflow run %s complete.", run.info.run_id)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-data-path", type=Path, default=None)
    parser.add_argument("--production-log-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--train-finetuned", action="store_true")
    parser.add_argument(
        "--embedder-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--finetuned-model-name", type=str, default="distilbert-base-uncased"
    )
    parser.add_argument("--lr-c", type=float, default=1.0, dest="lr_C")
    parser.add_argument("--finetuned-epochs", type=int, default=3)
    parser.add_argument("--finetuned-batch-size", type=int, default=16)
    parser.add_argument("--finetuned-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument(
        "--mlflow-experiment-name", type=str, default="query-relevance-classifier"
    )
    parser.add_argument("--no-cascade-eval", action="store_true")
    parser.add_argument(
        "--nli-model-name", type=str, default="cross-encoder/nli-deberta-v3-xsmall"
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        summary = run_training_pipeline(
            output_dir=args.output_dir,
            synthetic_data_path=args.synthetic_data_path,
            production_log_path=args.production_log_path,
            test_split=args.test_split,
            train_finetuned=args.train_finetuned,
            embedder_model_name=args.embedder_model_name,
            finetuned_model_name=args.finetuned_model_name,
            lr_C=args.lr_C,
            finetuned_epochs=args.finetuned_epochs,
            finetuned_batch_size=args.finetuned_batch_size,
            finetuned_learning_rate=args.finetuned_learning_rate,
            device=args.device,
            random_state=args.random_state,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            evaluate_full_cascade=not args.no_cascade_eval,
            nli_model_name=args.nli_model_name,
        )
    except RelevanceClassifierError as exc:
        logger.error("Training pipeline failed: %s: %s", type(exc).__name__, exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort catch for a CLI entry point
        logger.error("Training pipeline failed with an unexpected error: %s", exc)
        return 1

    print(f"\nRun complete. MLflow run_id={summary['run_id']}")
    print(
        "To promote a model version, use the MLflow UI/CLI once you're "
        "satisfied with this run's metrics (this script does not auto-promote)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
