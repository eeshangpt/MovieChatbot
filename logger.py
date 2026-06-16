import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "chatbot.log"

_FMT = "[%(asctime)s][%(levelname)-8s][%(name)s]:\t%(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _setup() -> None:
    root = logging.getLogger("chatbot")
    if root.handlers:
        return

    root.setLevel(_LOG_LEVEL)
    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    _LOG_DIR.mkdir(exist_ok=True)
    rotating = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    rotating.setFormatter(fmt)
    root.addHandler(rotating)


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(name)
