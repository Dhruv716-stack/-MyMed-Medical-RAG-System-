"""
self_rag.utils.helpers
======================

Compatibility facade for shared Self-RAG helpers.

This module preserves the original planned ``helpers.py`` layout while
re-exporting the stable utility functions already used across the
package. Keeping this facade avoids churn in the current integration
and lets future modules depend on a single helper surface.
"""

from __future__ import annotations

from .compat import accepts_parameter, call_with_supported_kwargs
from .documents import (
    average_reranker_score,
    build_citations,
    chunks_to_documents,
    deduplicate_documents,
    documents_to_chunks,
    normalize_score,
    pseudo_score_for_rank,
)
from .llm import (
    LLMInvocationError,
    ainvoke_llm,
    extract_json_payload,
    invoke_llm,
    normalize_llm_response,
)

__all__ = [
    "LLMInvocationError",
    "accepts_parameter",
    "ainvoke_llm",
    "average_reranker_score",
    "build_citations",
    "call_with_supported_kwargs",
    "chunks_to_documents",
    "deduplicate_documents",
    "documents_to_chunks",
    "extract_json_payload",
    "invoke_llm",
    "normalize_llm_response",
    "normalize_score",
    "pseudo_score_for_rank",
]
