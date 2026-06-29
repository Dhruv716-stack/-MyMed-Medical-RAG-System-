"""
Prompt templates used by the Self-RAG subsystem.
"""

from .retrieval_prompt import (
    SYSTEM_PROMPT,
    JSON_SCHEMA,
    build_retrieval_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "JSON_SCHEMA",
    "build_retrieval_prompt",
]