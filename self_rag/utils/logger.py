"""
self_rag.utils.logger
=====================

Lightweight logging helpers for the Self-RAG package.

The goal here is to provide a stable logger surface without forcing any
global logging configuration changes on the rest of the application.
"""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_SELF_RAG_LOGGER = "self_rag"


def get_self_rag_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Return a package-scoped logger.

    When ``name`` is omitted, the package root logger is returned.
    Passing a module name creates a child logger under ``self_rag``.
    """

    if not name:
        logger = logging.getLogger(DEFAULT_SELF_RAG_LOGGER)
    elif name.startswith(f"{DEFAULT_SELF_RAG_LOGGER}."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(
            f"{DEFAULT_SELF_RAG_LOGGER}.{name}"
        )

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def build_log_extra(
    **fields: Any,
) -> dict[str, Any]:
    """
    Build a sanitized ``extra`` payload for structured logging calls.
    """

    return {
        key: value
        for key, value in fields.items()
        if value is not None
    }


def safe_log_value(
    value: Any,
    *,
    max_length: int = 300,
) -> str:
    """
    Convert any value to a bounded log-safe string.
    """

    text = str(value).replace("\n", " ").strip()

    if len(text) <= max_length:
        return text

    return f"{text[:max_length].rstrip()}..."
