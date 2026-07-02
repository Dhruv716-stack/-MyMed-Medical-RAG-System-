"""
Prompt templates used by the Self-RAG subsystem.
"""

from .retrieval_prompt import (
    SYSTEM_PROMPT as RETRIEVAL_SYSTEM_PROMPT,
    JSON_SCHEMA as RETRIEVAL_JSON_SCHEMA,
    build_retrieval_prompt,
)

from .answer_prompt import (
    SYSTEM_PROMPT as ANSWER_SYSTEM_PROMPT,
    JSON_SCHEMA as ANSWER_JSON_SCHEMA,
    build_answer_prompt,
)

__all__ = [
    "RETRIEVAL_SYSTEM_PROMPT",
    "RETRIEVAL_JSON_SCHEMA",
    "build_retrieval_prompt",
    "ANSWER_SYSTEM_PROMPT",
    "ANSWER_JSON_SCHEMA",
    "build_answer_prompt",
]