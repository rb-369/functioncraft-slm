"""Structured logging utility with optional Rich console output."""

import logging
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def setup_logger(
    name: str = "functioncraft",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.handlers:
        return logger

    if _HAS_RICH:
        console = Console(file=sys.stdout)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(numeric_level)
        logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger
