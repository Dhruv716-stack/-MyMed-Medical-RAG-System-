"""
corrective_rag.crag_pipeline
=============================

Main Corrective RAG (CRAG) orchestrator.

Replaces the Self-RAG pipeline with a simpler, faster flow:

    1. Grade retrieved documents for relevance
    2. Based on grading:
       - ALL relevant   → pass docs through (metadata preserved)
       - SOME relevant  → keep relevant docs + supplement with web search
       - NONE relevant  → web search only (fallback to originals if no results)
    3. Return filtered docs with full metadata for downstream compression

Typical overhead: ~30-80ms vs Self-RAG's ~180-200ms.

NOTE: Knowledge refinement was removed because it stripped per-document
metadata (PDF source, page numbers) which is critical for citation
confidence in medical RAG. The existing contextual compression in
post_retrieval() handles compression while preserving metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from corrective_rag.document_grader import DocumentGrader, GradingResult
from tools.web_search import web_search

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CRAGResult:
    """Result of a Corrective RAG pass."""

    query: str
    refined_docs: List[Document] = field(default_factory=list)
    grading_result: Optional[GradingResult] = None
    web_search_used: bool = False
    web_search_docs: List[Document] = field(default_factory=list)
    action_taken: str = ""  # "all_relevant", "some_relevant", "none_relevant"

    # Timing
    grading_latency_ms: float = 0.0
    web_search_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ── CRAG Pipeline ────────────────────────────────────────────────────────────

class CRAGPipeline:
    """
    Corrective RAG pipeline.

    Usage
    -----
        crag = CRAGPipeline()
        result = crag.run(query="What is diabetes?", docs=retrieved_docs)
        # result.refined_docs → pass to post_retrieval for compression
    """

    def __init__(
        self,
        *,
        grader: DocumentGrader | None = None,
        web_search_max_results: int = 3,
    ) -> None:
        self._grader = grader or DocumentGrader()
        self._web_search_max_results = web_search_max_results

    def run(
        self,
        query: str,
        docs: List[Document],
    ) -> CRAGResult:
        """
        Execute the Corrective RAG flow.

        Parameters
        ----------
        query
            The user's question.
        docs
            Retrieved documents (already reranked).

        Returns
        -------
        CRAGResult
            Docs are returned with all metadata preserved (source, page, etc.).
            Downstream contextual compression handles any further distillation.
        """

        total_start = time.perf_counter()

        # ── Step 1: Grade documents ───────────────────────────────────────
        grade_start = time.perf_counter()
        grading = self._grader.grade(query=query, docs=docs)
        grading_latency = (time.perf_counter() - grade_start) * 1000.0

        logger.info(
            "CRAG Grading: %d relevant, %d irrelevant (%.1fms)",
            len(grading.relevant_docs),
            len(grading.irrelevant_docs),
            grading_latency,
        )

        # ── Step 2: Decide action & build context docs ────────────────────
        web_docs: List[Document] = []
        web_latency = 0.0
        web_used = False

        if grading.all_relevant:
            # All docs relevant — pass through with metadata intact
            action = "all_relevant"
            context_docs = grading.relevant_docs
            logger.info("CRAG: All docs relevant, no web search needed")

        elif grading.some_relevant:
            # Some relevant — supplement with web search
            action = "some_relevant"
            web_start = time.perf_counter()
            web_docs = web_search(
                query, max_results=self._web_search_max_results
            )
            web_latency = (time.perf_counter() - web_start) * 1000.0
            web_used = True
            context_docs = grading.relevant_docs + web_docs
            logger.info(
                "CRAG: Partial relevance — supplemented with %d web results (%.1fms)",
                len(web_docs),
                web_latency,
            )

        else:
            # None relevant — web search only
            action = "none_relevant"
            web_start = time.perf_counter()
            web_docs = web_search(
                query, max_results=self._web_search_max_results + 2
            )
            web_latency = (time.perf_counter() - web_start) * 1000.0
            web_used = True

            if web_docs:
                context_docs = web_docs
            else:
                # Web search also failed — fall back to original docs
                # (better than nothing)
                logger.warning(
                    "CRAG: No relevant docs and web search returned nothing; "
                    "falling back to original docs"
                )
                context_docs = docs

            logger.info(
                "CRAG: No relevant docs — using %d web results (%.1fms)",
                len(web_docs),
                web_latency,
            )

        # ── Step 3: Return docs with metadata preserved ───────────────────
        # No knowledge refinement — metadata (PDF source, page numbers)
        # is preserved intact for citation in generation.
        # Contextual compression in post_retrieval() will handle
        # any further distillation while keeping metadata.

        total_latency = (time.perf_counter() - total_start) * 1000.0

        logger.info(
            "CRAG complete: action=%s, docs=%d, total=%.1fms",
            action,
            len(context_docs),
            total_latency,
        )

        return CRAGResult(
            query=query,
            refined_docs=context_docs,
            grading_result=grading,
            web_search_used=web_used,
            web_search_docs=web_docs,
            action_taken=action,
            grading_latency_ms=grading_latency,
            web_search_latency_ms=web_latency,
            total_latency_ms=total_latency,
        )
