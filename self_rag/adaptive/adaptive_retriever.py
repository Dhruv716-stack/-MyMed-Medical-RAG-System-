"""
self_rag.adaptive.adaptive_retriever
====================================

Adaptive retriever that loops retrieval, reranking, and reflection
without changing the existing retriever modules outside ``self_rag``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any

from langchain_core.documents import Document

from self_rag.adaptive.retry_strategy import RetryStrategy
from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.reflection.retrieval_reflection import (
    RetrievalReflectionEngine,
)
from self_rag.schemas.retrieval import RetrievalReflectionOutput, RetrievalRequest
from self_rag.utils.compat import call_with_supported_kwargs
from self_rag.utils.documents import (
    deduplicate_documents,
    documents_to_chunks,
    normalize_score,
    pseudo_score_for_rank,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdaptiveRetrieverResponse:
    """
    Runtime output of the adaptive retriever.
    """

    documents: list[Document]
    reflection_output: RetrievalReflectionOutput
    history: list[dict[str, Any]] = field(default_factory=list)


class AdaptiveRetriever:
    """
    Orchestrate retrieval, reranking, and retrieval reflection.
    """

    def __init__(
        self,
        *,
        hybrid_retriever: Any,
        mmr_retriever_factory: Any,
        reranker: Any,
        retrieval_reflection: RetrievalReflectionEngine,
        retry_strategy: RetryStrategy | None = None,
        reranker_scorer: Any | None = None,
        settings: SelfRAGSettings | None = None,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._mmr_retriever_factory = mmr_retriever_factory
        self._reranker = reranker
        self._reranker_scorer = reranker_scorer
        self._retrieval_reflection = retrieval_reflection
        self._settings = settings or get_self_rag_settings()
        self._retry_strategy = retry_strategy or RetryStrategy(
            settings=self._settings
        )

    def retrieve(
        self,
        *,
        query: str,
        rewritten_query: str | None = None,
        queries: list[str] | None = None,
        docs: list[Document] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        restrict_to_user_upload: bool = False,
        top_k: int | None = None,
    ) -> AdaptiveRetrieverResponse:
        """
        Run the adaptive retrieval loop.
        """

        retrieval_queries = self._build_search_queries(
            query=query,
            rewritten_query=rewritten_query,
            queries=queries,
        )

        current_top_k = top_k or self._settings.INITIAL_TOP_K
        knowledge_docs = list(docs or [])

        total_retrieval_latency_ms = 0.0
        total_reranker_latency_ms = 0.0
        total_reflection_latency_ms = 0.0
        history: list[dict[str, Any]] = []

        last_documents: list[Document] = []
        final_output: RetrievalReflectionOutput | None = None

        for iteration in range(1, self._settings.MAX_RETRIEVAL_ITERATIONS + 1):
            iteration_retrieval_start = time.perf_counter()

            hybrid_docs = self._run_hybrid_retrieval(
                queries=retrieval_queries,
                docs=knowledge_docs,
                current_top_k=current_top_k,
                user_id=user_id,
                session_id=session_id,
                restrict_to_user_upload=restrict_to_user_upload,
            )
            mmr_docs = self._run_mmr_retrieval(
                queries=retrieval_queries,
                current_top_k=current_top_k,
                user_id=user_id,
                session_id=session_id,
                restrict_to_user_upload=restrict_to_user_upload,
            )
            retrieval_latency_ms = (
                time.perf_counter() - iteration_retrieval_start
            ) * 1000.0
            total_retrieval_latency_ms += retrieval_latency_ms

            combined_documents = deduplicate_documents(hybrid_docs + mmr_docs)

            reranker_start = time.perf_counter()
            reranked_documents = self._run_reranker(
                query=rewritten_query or query,
                docs=combined_documents,
                top_k=current_top_k,
            )
            reranker_scores = self._score_documents(
                query=rewritten_query or query,
                docs=reranked_documents,
            )
            reranker_latency_ms = (
                time.perf_counter() - reranker_start
            ) * 1000.0
            total_reranker_latency_ms += reranker_latency_ms

            retrieved_chunks = documents_to_chunks(
                reranked_documents,
                reranker_scores=reranker_scores,
            )

            reflection_request = RetrievalRequest(
                query=query,
                rewritten_query=rewritten_query,
                top_k=current_top_k,
                iteration=iteration,
                retrieved_chunks=retrieved_chunks,
            )

            reflection_output = self._retrieval_reflection.reflect(
                reflection_request,
                retrieval_latency_ms=retrieval_latency_ms,
                reranker_latency_ms=reranker_latency_ms,
                total_iterations=iteration,
                final_top_k=current_top_k,
            )
            total_reflection_latency_ms += (
                reflection_output.metrics.reflection_latency_ms
            )

            decision = reflection_output.retrieval_result.retrieval_decision
            history.append(
                {
                    "iteration": iteration,
                    "top_k": current_top_k,
                    "retrieved_documents": len(reranked_documents),
                    "retrieval_confidence": decision.retrieval_confidence,
                    "sufficient_context": decision.sufficient_context,
                    "reasoning": decision.reasoning,
                }
            )

            last_documents = reranked_documents
            final_output = reflection_output

            if not self._retry_strategy.should_retry_retrieval(
                decision,
                iteration,
            ):
                break

            current_top_k = self._retry_strategy.next_top_k(
                current_top_k,
                decision,
            )

        if final_output is None:
            empty_request = RetrievalRequest(
                query=query,
                rewritten_query=rewritten_query,
                top_k=current_top_k,
                iteration=1,
                retrieved_chunks=[],
            )
            final_output = self._retrieval_reflection.reflect(
                empty_request,
                retrieval_latency_ms=0.0,
                reranker_latency_ms=0.0,
                total_iterations=1,
                final_top_k=current_top_k,
            )

        final_output.metrics.retrieval_latency_ms = total_retrieval_latency_ms
        final_output.metrics.reranker_latency_ms = total_reranker_latency_ms
        final_output.metrics.reflection_latency_ms = total_reflection_latency_ms
        final_output.retrieval_result.iterations = len(history) or 1
        final_output.retrieval_result.final_top_k = current_top_k
        final_output.success = (
            final_output.retrieval_result.retrieval_decision.sufficient_context
        )
        final_output.metadata["history"] = history
        final_output.metadata["search_queries"] = retrieval_queries

        return AdaptiveRetrieverResponse(
            documents=last_documents,
            reflection_output=final_output,
            history=history,
        )

    def _build_search_queries(
        self,
        *,
        query: str,
        rewritten_query: str | None,
        queries: list[str] | None,
    ) -> list[str]:
        """
        Deduplicate retrieval queries while preserving priority order.
        """

        ordered_queries = [
            rewritten_query or query,
            *(queries or []),
            query,
        ]

        seen: set[str] = set()
        result: list[str] = []

        for candidate in ordered_queries:
            normalized = (candidate or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(normalized)

        return result or [query]

    def _run_hybrid_retrieval(
        self,
        *,
        queries: list[str],
        docs: list[Document],
        current_top_k: int,
        user_id: str | None,
        session_id: str | None,
        restrict_to_user_upload: bool,
    ) -> list[Document]:
        """
        Run hybrid retrieval for each query variant.
        """

        results: list[Document] = []

        for candidate_query in queries:
            try:
                retrieved = call_with_supported_kwargs(
                    self._hybrid_retriever,
                    query=candidate_query,
                    docs=docs,
                    top_k=current_top_k,
                    k=current_top_k,
                    user_id=user_id,
                    session_id=session_id,
                    restrict_to_user_upload=restrict_to_user_upload,
                )
                results.extend(list(retrieved or []))
            except Exception as exc:
                logger.warning(
                    "Hybrid retrieval failed for query '%s': %s",
                    candidate_query,
                    exc,
                )

        return results

    def _run_mmr_retrieval(
        self,
        *,
        queries: list[str],
        current_top_k: int,
        user_id: str | None,
        session_id: str | None,
        restrict_to_user_upload: bool,
    ) -> list[Document]:
        """
        Run MMR retrieval for each query variant.
        """

        try:
            retriever = call_with_supported_kwargs(
                self._mmr_retriever_factory,
                top_k=current_top_k,
                k=current_top_k,
                fetch_k=max(current_top_k * 2, current_top_k),
                user_id=user_id,
                session_id=session_id,
                restrict_to_user_upload=restrict_to_user_upload,
            )
        except Exception as exc:
            logger.warning("MMR retriever construction failed: %s", exc)
            return []

        results: list[Document] = []
        for candidate_query in queries:
            try:
                mmr_results = retriever.invoke(candidate_query)
                results.extend(list(mmr_results or []))
            except Exception as exc:
                logger.warning(
                    "MMR retrieval failed for query '%s': %s",
                    candidate_query,
                    exc,
                )

        return results

    def _run_reranker(
        self,
        *,
        query: str,
        docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        Rerank combined retrieval results.
        """

        if not docs:
            return []

        try:
            ranked = call_with_supported_kwargs(
                self._reranker,
                query=query,
                docs=docs,
                top_k=top_k,
                k=top_k,
            )
            return list(ranked or [])
        except Exception as exc:
            logger.warning(
                "Reranker failed; using truncated retrieval order: %s",
                exc,
            )
            return list(docs[:top_k])

    def _score_documents(
        self,
        *,
        query: str,
        docs: list[Document],
    ) -> list[float]:
        """
        Score reranked documents for reflection. Falls back to
        rank-based pseudo-scores when a scorer is unavailable.
        """

        if not docs:
            return []

        if self._reranker_scorer is not None:
            try:
                scores = call_with_supported_kwargs(
                    self._reranker_scorer,
                    query=query,
                    docs=docs,
                    documents=docs,
                )
                normalized = [
                    normalize_score(score)
                    for score in list(scores or [])
                ]
                if len(normalized) == len(docs):
                    return normalized
            except Exception as exc:
                logger.warning(
                    "Reranker scorer unavailable; using pseudo-scores: %s",
                    exc,
                )

        return [
            pseudo_score_for_rank(index, len(docs))
            for index in range(1, len(docs) + 1)
        ]
