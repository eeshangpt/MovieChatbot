"""Shared utilities used across the package: device resolution and logging."""

from __future__ import annotations

import logging
import sys

from .exceptions import ConfigError


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve the torch device to use, with graceful fallback.

    Args:
        requested_device: "auto", "cuda", or "cpu".

    Returns:
        "cuda" or "cpu".

    Raises:
        ConfigError: if requested_device is an unrecognized value, or if
            "cuda" was explicitly requested but is not available / torch
            is not installed.
    """
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ConfigError(
            f"Invalid device '{requested_device}'. Expected one of: auto, cuda, cpu."
        )

    try:
        import torch
    except ImportError as exc:
        if requested_device == "cuda":
            raise ConfigError("CUDA was requested but torch is not installed.") from exc
        return "cpu"
    except Exception as exc:  # noqa: BLE001
        # torch can fail to import for reasons other than "not installed"
        # (e.g. a broken/incomplete wheel raising OSError while loading its
        # native shared libraries). Treat any such failure as "torch is
        # unusable here" rather than crashing resolve_device with an
        # unrelated traceback.
        if requested_device == "cuda":
            raise ConfigError(
                f"CUDA was requested but torch failed to import: {exc}"
            ) from exc
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if requested_device == "cuda" and not cuda_available:
        raise ConfigError(
            "CUDA was explicitly requested but no CUDA-capable GPU is available."
        )

    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"

    return requested_device


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module-level logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
