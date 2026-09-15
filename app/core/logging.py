"""
Application-wide logging configuration.

Called once at startup (in app/main.py). Every other module just does:

    import logging
    logger = logging.getLogger(__name__)

and inherits this configuration — no per-module setup needed.
"""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. during tests that spin up the app repeatedly).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers unless we're debugging.
    if log_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Thin convenience wrapper so call sites can do:

        from app.core.logging import get_logger
        logger = get_logger(__name__)

    instead of importing the stdlib `logging` module directly everywhere.
    Relies on `configure_logging()` having been called once at app startup —
    this function does not configure handlers itself.
    """
    return logging.getLogger(name)
