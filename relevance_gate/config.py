"""Central configuration for the query relevance classification system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigError
from .utils import resolve_device


@dataclass
class Tier1Config:
    """Config for the Tier 1 ensemble (embedding+LR, optionally + fine-tuned)."""

    embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    finetuned_model_name: str = "distilbert-base-uncased"
    lr_model_path: Path = Path("artifacts/tier1_lr")
    ft_model_path: Path = Path("artifacts/tier1_finetuned")

    # Ensemble weights. w_ft starts at 0 until a fine-tuned model exists
    # and has been validated against held-out production data.
    w_lr: float = 1.0
    w_ft: float = 0.0

    # Score bands: score >= relevant_threshold -> RELEVANT,
    # score <= out_of_domain_threshold -> OUT_OF_DOMAIN, else uncertain.
    relevant_threshold: float = 0.65
    out_of_domain_threshold: float = 0.35

    # If |lr_prob - ft_prob| exceeds this, escalate to Tier 2 even if the
    # weighted ensemble score looks confident (query-by-committee signal).
    disagreement_threshold: float = 0.4

    embedding_batch_size: int = 64

    def __post_init__(self) -> None:
        if not 0.0 <= self.out_of_domain_threshold < self.relevant_threshold <= 1.0:
            raise ConfigError(
                "Tier1Config requires 0 <= out_of_domain_threshold < "
                f"relevant_threshold <= 1, got out_of_domain_threshold="
                f"{self.out_of_domain_threshold}, relevant_threshold="
                f"{self.relevant_threshold}."
            )
        if self.w_lr < 0 or self.w_ft < 0:
            raise ConfigError("Tier1Config ensemble weights must be non-negative.")
        if self.w_lr + self.w_ft == 0:
            raise ConfigError("Tier1Config ensemble weights cannot both be zero.")
        if not 0.0 < self.disagreement_threshold <= 1.0:
            raise ConfigError("Tier1Config.disagreement_threshold must be in (0, 1].")
        if self.embedding_batch_size < 1:
            raise ConfigError("Tier1Config.embedding_batch_size must be >= 1.")


@dataclass
class Tier2Config:
    """Config for the Tier 2 zero-shot NLI fallback classifier."""

    nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall"
    hypothesis_template: str = "This query is about {}."
    in_domain_label: str = "movies or TV shows"
    out_domain_label: str = "something other than movies or TV shows"

    relevant_threshold: float = 0.6
    out_of_domain_threshold: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.out_of_domain_threshold < self.relevant_threshold <= 1.0:
            raise ConfigError(
                "Tier2Config requires 0 <= out_of_domain_threshold < "
                f"relevant_threshold <= 1, got out_of_domain_threshold="
                f"{self.out_of_domain_threshold}, relevant_threshold="
                f"{self.relevant_threshold}."
            )


@dataclass
class FeedbackConfig:
    """Controls the production data collection / exploration sampling loop."""

    # Fraction of OUT_OF_DOMAIN decisions that are sent through to the LLM
    # anyway, purely to collect ground-truth labels and detect drift on
    # queries that would otherwise never be observed again.
    exploration_rate: float = 0.05
    log_path: Path = Path("logs/query_log.jsonl")

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ConfigError(
                f"FeedbackConfig.exploration_rate must be in [0, 1], got {self.exploration_rate}."
            )


@dataclass
class MLflowConfig:
    """Config for the (manually-triggered) MLflow-tracked training pipeline.

    NOTE: MLflow 3.x put the plain filesystem backend ("file:./mlruns") into
    maintenance mode and raises by default unless MLFLOW_ALLOW_FILE_STORE=true
    is set. sqlite is the currently-recommended lightweight local backend, so
    that's the default here.
    """

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "query-relevance-classifier"


@dataclass
class RelevanceClassifierConfig:
    """Top-level config bundling all sub-configs."""

    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # "auto" picks cuda if available, else cpu.
    device: str = "auto"

    # What to do when Tier 2 is also uncertain: "relevant" or "out_of_domain".
    default_policy_label: str = "relevant"

    def __post_init__(self) -> None:
        if self.default_policy_label not in {"relevant", "out_of_domain"}:
            raise ConfigError(
                "default_policy_label must be 'relevant' or 'out_of_domain', "
                f"got '{self.default_policy_label}'."
            )
        # Validate device early so failures surface at config time, not
        # buried inside a model-loading call much later.
        self.resolved_device = resolve_device(self.device)
