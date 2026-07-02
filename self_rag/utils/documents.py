"""
self_rag.utils.documents
========================

Document conversion helpers for bridging the existing LangChain-based
retrieval pipeline with Self-RAG schemas.
"""

from __future__ import annotations

import math
import re
from typing import Any

from langchain_core.documents import Document

from self_rag.schemas.answer import Citation
from self_rag.schemas.retrieval import RetrievedChunk

_CITATION_PATTERN = re.compile(
    r"\[Source:\s*(?P<source>[^|\]]+?)"
    r"(?:\s*\|\s*Page:\s*(?P<page>[^\]]+))?\]",
    flags=re.IGNORECASE,
)


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Deduplicate documents by normalized text while preserving order.
    """

    seen: set[str] = set()
    unique_documents: list[Document] = []

    for document in documents:
        text = (document.page_content or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_documents.append(document)

    return unique_documents


def normalize_score(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    """
    Normalize raw scores into a 0-1 range.

    Cross-encoder rerankers often emit unbounded logits, so those are
    converted with a sigmoid transform.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(score) or math.isinf(score):
        return default

    if 0.0 <= score <= 1.0:
        return score

    clamped = max(min(score, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-clamped))


def pseudo_score_for_rank(
    rank: int,
    total: int,
) -> float:
    """
    Generate a stable fallback score when a model score is unavailable.
    """

    if total <= 0:
        return 0.0

    return max(0.05, 1.0 - ((rank - 1) / max(total, 1)))


def documents_to_chunks(
    documents: list[Document],
    *,
    reranker_scores: list[float] | None = None,
    retrieval_scores: list[float] | None = None,
) -> list[RetrievedChunk]:
    """
    Convert LangChain documents into Self-RAG chunks.
    """

    chunks: list[RetrievedChunk] = []
    total = len(documents)

    for index, document in enumerate(documents, start=1):
        metadata = dict(document.metadata or {})
        reranker_score = (
            reranker_scores[index - 1]
            if reranker_scores and index - 1 < len(reranker_scores)
            else metadata.get("reranker_score")
        )
        retrieval_score = (
            retrieval_scores[index - 1]
            if retrieval_scores and index - 1 < len(retrieval_scores)
            else metadata.get("retrieval_score", metadata.get("score"))
        )

        document_id = _coerce_identifier(
            metadata.get("document_id")
            or metadata.get("doc_id")
            or metadata.get("source")
            or f"document-{index}"
        )
        chunk_id = _coerce_identifier(
            metadata.get("chunk_id")
            or metadata.get("id")
            or f"{document_id}-chunk-{index}"
        )

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=document.page_content,
                metadata=metadata,
                retrieval_score=normalize_score(
                    retrieval_score,
                    default=pseudo_score_for_rank(index, total),
                ),
                reranker_score=normalize_score(
                    reranker_score,
                    default=pseudo_score_for_rank(index, total),
                ),
                rank=index,
            )
        )

    return chunks


def chunks_to_documents(
    chunks: list[RetrievedChunk],
) -> list[Document]:
    """
    Convert Self-RAG chunks back into LangChain documents.
    """

    documents: list[Document] = []

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.setdefault("chunk_id", chunk.chunk_id)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("retrieval_score", chunk.retrieval_score)
        metadata.setdefault("reranker_score", chunk.reranker_score)

        documents.append(
            Document(
                page_content=chunk.text,
                metadata=metadata,
            )
        )

    return documents


def average_reranker_score(
    chunks: list[RetrievedChunk],
) -> float:
    """
    Compute average normalized reranker score.
    """

    if not chunks:
        return 0.0

    return sum(chunk.reranker_score for chunk in chunks) / len(chunks)


def build_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """
    Build structured citations from an answer and retrieved chunks.
    """

    chunk_lookup = {
        (
            str(chunk.metadata.get("source", "")).strip().lower(),
            _normalize_page_value(chunk.metadata.get("page")),
        ): chunk
        for chunk in chunks
    }

    citations: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()

    for match in _CITATION_PATTERN.finditer(answer):
        source = match.group("source").strip()
        page = _parse_page(match.group("page"))
        key = (source.lower(), page)

        if key in seen:
            continue
        seen.add(key)

        matched_chunk = chunk_lookup.get(key)

        citations.append(
            Citation(
                document_id=(
                    matched_chunk.document_id
                    if matched_chunk
                    else source
                ),
                chunk_id=(
                    matched_chunk.chunk_id
                    if matched_chunk
                    else f"{source}-{page or 'na'}"
                ),
                source=source,
                page=page,
                score=(
                    matched_chunk.reranker_score
                    if matched_chunk
                    else 0.0
                ),
            )
        )

    if citations:
        return citations

    fallback: list[Citation] = []
    for chunk in chunks[:3]:
        fallback.append(
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                source=str(chunk.metadata.get("source", chunk.document_id)),
                page=_parse_page(chunk.metadata.get("page")),
                score=chunk.reranker_score,
            )
        )

    return fallback


def _parse_page(value: Any) -> int | None:
    """
    Convert a page-like value to an integer when possible.
    """

    if value in (None, "", "N/A"):
        return None

    try:
        page = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return page if page > 0 else None


def _normalize_page_value(value: Any) -> int | None:
    """
    Normalize page metadata for dictionary lookups.
    """

    return _parse_page(value)


def _coerce_identifier(value: Any) -> str:
    """
    Return a stable string identifier.
    """

    return str(value).strip() or "unknown"
