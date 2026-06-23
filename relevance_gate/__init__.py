"""relevance_gate: a cheap pre-filter that classifies user queries as
relevant (movie/TV) or out-of-domain before routing to an expensive LLM.
"""

from .config import (
    FeedbackConfig,
    MLflowConfig,
    RelevanceClassifierConfig,
    Tier1Config,
    Tier2Config,
)
from .data_generation import (
    DEFAULT_NEGATIVE_CATEGORIES,
    DEFAULT_POSITIVE_CATEGORIES,
    generate_synthetic_dataset,
)
from .data_io import load_training_examples, save_training_examples
from .embedding_classifier import EmbeddingLRClassifier
from .evaluation import (
    CascadeEvalMetrics,
    EvalMetrics,
    evaluate_classifier,
    evaluate_probabilistic,
)
from .exceptions import (
    ConfigError,
    DataGenerationError,
    EmbeddingError,
    InferenceError,
    InvalidInputError,
    LabelDerivationError,
    ModelLoadError,
    ModelNotFittedError,
    PersistenceError,
    RelevanceClassifierError,
    TrainingError,
)
from .finetuned_classifier import FineTunedClassifier
from .label_derivation import (
    derive_label,
    derive_label_heuristic,
    derive_label_llm_judge,
)
from .query_logger import QueryLogger
from .relevance_classifier import QueryRelevanceClassifier
from .schemas import (
    ClassificationResult,
    Label,
    LabelSource,
    TierStage,
    TrainingExample,
)
from .tier1_ensemble import Tier1Ensemble, Tier1Result
from .train_pipeline import run_training_pipeline
from .zero_shot_classifier import ZeroShotRelevanceChecker

__all__ = [
    "RelevanceClassifierConfig",
    "Tier1Config",
    "Tier2Config",
    "FeedbackConfig",
    "MLflowConfig",
    "Label",
    "LabelSource",
    "TierStage",
    "ClassificationResult",
    "TrainingExample",
    "EmbeddingLRClassifier",
    "FineTunedClassifier",
    "ZeroShotRelevanceChecker",
    "Tier1Ensemble",
    "Tier1Result",
    "QueryRelevanceClassifier",
    "generate_synthetic_dataset",
    "DEFAULT_POSITIVE_CATEGORIES",
    "DEFAULT_NEGATIVE_CATEGORIES",
    "save_training_examples",
    "load_training_examples",
    "derive_label",
    "derive_label_heuristic",
    "derive_label_llm_judge",
    "QueryLogger",
    "EvalMetrics",
    "CascadeEvalMetrics",
    "evaluate_probabilistic",
    "evaluate_classifier",
    "run_training_pipeline",
    "RelevanceClassifierError",
    "ConfigError",
    "ModelLoadError",
    "ModelNotFittedError",
    "EmbeddingError",
    "InferenceError",
    "InvalidInputError",
    "DataGenerationError",
    "LabelDerivationError",
    "PersistenceError",
    "TrainingError",
]
