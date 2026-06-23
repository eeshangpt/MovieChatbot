"""Custom exception hierarchy for the query relevance classification system.

Using specific exception types (rather than bare Exception) lets calling
code distinguish between configuration errors, model loading failures,
inference-time failures, and data problems -- each of which usually
needs a different recovery strategy in production (e.g. retry vs. fail
fast vs. fall back to a default policy).
"""

from __future__ import annotations


class RelevanceClassifierError(Exception):
    """Base exception for all errors raised by this package."""


class ConfigError(RelevanceClassifierError):
    """Raised when configuration values are missing, malformed, or inconsistent."""


class ModelLoadError(RelevanceClassifierError):
    """Raised when a model (embedder, classifier head, NLI model, etc.) fails to load."""


class ModelNotFittedError(RelevanceClassifierError):
    """Raised when inference is attempted on a classifier that hasn't been trained/loaded."""


class EmbeddingError(RelevanceClassifierError):
    """Raised when text embedding fails (e.g. CUDA OOM, tokenization failure)."""


class InferenceError(RelevanceClassifierError):
    """Raised when a model fails during the forward/inference pass."""


class TrainingError(RelevanceClassifierError):
    """Raised when a model training/fine-tuning run fails."""


class TrainingError(RelevanceClassifierError):
    """Raised when fine-tuning a Tier 1b classifier fails (setup, OOM, or
    a degenerate training run such as NaN loss)."""


class InvalidInputError(RelevanceClassifierError):
    """Raised when input data (queries, labels, training examples) is invalid."""


class DataGenerationError(RelevanceClassifierError):
    """Raised when synthetic data generation via an LLM fails."""


class LabelDerivationError(RelevanceClassifierError):
    """Raised when deriving a label from an LLM response fails."""


class PersistenceError(RelevanceClassifierError):
    """Raised when saving/loading model artifacts to/from disk fails."""
