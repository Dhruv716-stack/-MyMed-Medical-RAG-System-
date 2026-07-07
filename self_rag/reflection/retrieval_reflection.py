"""
self_rag.reflection.retrieval_reflection
========================================

Retrieval reflection engine that combines LLM-based criticism with a
safe heuristic fallback.
"""

from __future__ import annotations

import logging
import time

from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.critics.retrieval_critic import (
    RetrievalCritic,
    RetrievalCriticError,
)
from self_rag.schemas.retrieval import (
    AdaptiveRetrievalResult,
    RetrievalDecision,
    RetrievalMetrics,
    RetrievalReflectionOutput,
    RetrievalRequest,
)
from self_rag.utils.documents import average_reranker_score

logger = logging.getLogger(__name__)


class RetrievalReflectionEngine:
    """
    Evaluate retrieved evidence and decide whether it is sufficient.
    """

    def __init__(
        self,
        *,
        critic: RetrievalCritic | None = None,
        settings: SelfRAGSettings | None = None,
    ) -> None:
        self._critic = critic
        self._settings = settings or get_self_rag_settings()

    def reflect(
        self,
        request: RetrievalRequest,
        *,
        retrieval_latency_ms: float = 0.0,
        reranker_latency_ms: float = 0.0,
        total_iterations: int | None = None,
        final_top_k: int | None = None,
    ) -> RetrievalReflectionOutput:
        """
        Reflect on retrieval quality and produce a structured decision.
        """

        reflection_start = time.perf_counter()
        evaluation_source = "heuristic"
        fallback_error: str | None = None

        if (
            self._settings.ENABLE_RETRIEVAL_REFLECTION
            and self._critic is not None
        ):
            try:
                decision = self._critic.evaluate_sync(request)
                evaluation_source = "critic"
            except RetrievalCriticError as exc:
                fallback_error = str(exc)
                logger.warning(
                    "Retrieval critic failed; falling back to heuristic reflection: %s",
                    exc,
                )
                decision = self._heuristic_decision(request)
                evaluation_source = "heuristic_fallback"
        else:
            decision = self._heuristic_decision(request)

        reflection_latency_ms = (
            time.perf_counter() - reflection_start
        ) * 1000.0

        metrics = RetrievalMetrics(
            retrieval_latency_ms=retrieval_latency_ms,
            reranker_latency_ms=reranker_latency_ms,
            reflection_latency_ms=reflection_latency_ms,
            retrieved_document_count=len(request.retrieved_chunks),
            average_reranker_score=average_reranker_score(
                request.retrieved_chunks
            ),
        )

        result = AdaptiveRetrievalResult(
            iterations=total_iterations or request.iteration,
            final_top_k=final_top_k or request.top_k,
            retrieval_decision=decision,
            retrieved_chunks=request.retrieved_chunks,
        )

        metadata = {
            "evaluation_source": evaluation_source,
            "iteration": request.iteration,
            "requested_top_k": request.top_k,
            "rewritten_query": request.rewritten_query,
        }

        if fallback_error is not None:
            metadata["fallback_error"] = fallback_error

        return RetrievalReflectionOutput(
            success=decision.sufficient_context,
            retrieval_result=result,
            metrics=metrics,
            metadata=metadata,
        )

    def _heuristic_decision(
        self,
        request: RetrievalRequest,
    ) -> RetrievalDecision:
        """
        Produce a bounded heuristic decision when the critic is disabled
        or unavailable.
        """

        chunks = request.retrieved_chunks
        document_count = len(chunks)

        if not chunks:
            return RetrievalDecision(
                sufficient_context=False,
                retrieval_confidence=0.0,
                suggested_top_k=min(
                    50,
                    request.top_k + self._settings.TOP_K_INCREMENT,
                ),
                reasoning=(
                    "No documents were retrieved, so retrieval should be retried "
                    "with a larger candidate set."
                ),
                reflection_metadata={
                    "relevance": 0.0,
                    "coverage": 0.0,
                    "reranker_quality": 0.0,
                    "missing_topics": ["No supporting context retrieved."],
                },
            )

        avg_reranker = average_reranker_score(chunks)
        avg_retrieval = sum(
            chunk.retrieval_score for chunk in chunks
        ) / document_count
        context_chars = sum(
            len(chunk.text.strip()) for chunk in chunks[:4]
        )
        coverage = min(1.0, context_chars / 1600.0)
        volume = min(1.0, document_count / max(3, request.top_k))

        retrieval_confidence = min(
            1.0,
            (
                avg_reranker * 0.45
                + avg_retrieval * 0.25
                + coverage * 0.20
                + volume * 0.10
            ),
        )

        sufficient_context = (
            document_count >= min(3, request.top_k)
            and avg_reranker
            >= max(
                0.35,
                self._settings.MIN_RERANKER_SCORE * 0.80,
            )
            and coverage >= 0.35
        )

        suggested_top_k = (
            request.top_k
            if sufficient_context
            else min(
                50,
                request.top_k + self._settings.TOP_K_INCREMENT,
            )
        )

        reasoning = (
            f"Retrieved {document_count} chunks with average reranker "
            f"score {avg_reranker:.2f} and coverage {coverage:.2f}."
        )

        return RetrievalDecision(
            sufficient_context=sufficient_context,
            retrieval_confidence=retrieval_confidence,
            suggested_top_k=suggested_top_k,
            reasoning=reasoning,
            reflection_metadata={
                "relevance": avg_retrieval,
                "coverage": coverage,
                "reranker_quality": avg_reranker,
                "missing_topics": [],
            },
        )
