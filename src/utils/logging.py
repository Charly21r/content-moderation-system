"""Structured logging setup for the content moderation system.

Usage:
    from utils.logging import setup_logging
    setup_logging()  # call once at entry point (main scripts)
"""

import logging
import sys
from typing import Literal


def setup_logging(
    level: int = logging.INFO,
    fmt: Literal["json", "text"] = "text",
) -> None:
    """Configure root logger with structured output.

    Args:
        level: Logging level (default INFO).
        fmt: "text" for human-readable, "json" for machine-parseable.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
