"""
self_rag.pipeline.self_rag_pipeline
===================================

Top-level Self-RAG orchestration layer that integrates with the
existing MyMED retrieval and generation stack without modifying code
outside the ``self_rag`` package.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document

from self_rag.adaptive import AdaptiveRetriever, RetryStrategy
from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.critics import AnswerCritic, ConfidenceEngine, RetrievalCritic
from self_rag.reflection import (
    AnswerReflectionEngine,
    RetrievalReflectionEngine,
)
from self_rag.schemas.answer import (
    AnswerEvaluation,
    AnswerMetrics,
    AnswerReflectionOutput,
    GeneratedAnswer,
)
from self_rag.schemas.confidence import (
    ConfidenceBreakdown,
    ConfidenceComponent,
    ConfidenceEngineOutput,
    ConfidenceLevel,
    ConfidenceMetrics,
    ConfidenceResult,
)
from self_rag.schemas.pipeline import (
    PipelineMetrics,
    SelfRAGPipelineResult,
)
from self_rag.schemas.retrieval import (
    AdaptiveRetrievalResult,
    RetrievalDecision,
    RetrievalMetrics,
    RetrievalReflectionOutput,
)
from self_rag.utils.compat import call_with_supported_kwargs
from self_rag.utils.documents import (
    documents_to_chunks,
    normalize_score,
)

logger = logging.getLogger(__name__)


class SelfRAGPipeline:
    """
    End-to-end Self-RAG pipeline.

    By default this class lazy-loads the project's existing query
    rewriter, retrievers, compressor, and generator so it can be
    adopted without edits outside ``self_rag``.
    """

    def __init__(
        self,
        *,
        settings: SelfRAGSettings | None = None,
        llm: Any | None = None,
        query_rewriter: Any | None = None,
        router: Any | None = None,
        multi_query_generator: Any | None = None,
        hybrid_retriever: Any | None = None,
        mmr_retriever_factory: Any | None = None,
        reranker: Any | None = None,
        reranker_scorer: Any | None = None,
        compressor: Any | None = None,
        generator: Any | None = None,
        answer_cleaner: Any | None = None,
        retrieval_critic: RetrievalCritic | None = None,
        answer_critic: AnswerCritic | None = None,
        confidence_engine: ConfidenceEngine | None = None,
    ) -> None:
        self._settings = settings or get_self_rag_settings()
        self._llm = llm
        self._query_rewriter = query_rewriter
        self._router = router
        self._multi_query_generator = multi_query_generator
        self._hybrid_retriever = hybrid_retriever
        self._mmr_retriever_factory = mmr_retriever_factory
        self._reranker = reranker
        self._reranker_scorer = reranker_scorer
        self._compressor = compressor
        self._generator = generator
        self._answer_cleaner = answer_cleaner
        self._retrieval_critic = retrieval_critic
        self._answer_critic = answer_critic
        self._confidence_engine = confidence_engine
        self._retry_strategy = RetryStrategy(settings=self._settings)
        self._adaptive_retriever: AdaptiveRetriever | None = None
        self._answer_reflection: AnswerReflectionEngine | None = None

    def run(
        self,
        query: str,
        *,
        docs: list[Document] | None = None,
        queries: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        restrict_to_user_upload: bool = False,
        top_k: int | None = None,
        expand_queries: bool = False,
    ) -> SelfRAGPipelineResult:
        """
        Execute the Self-RAG pipeline for a single query.
        """

        total_start = time.perf_counter()
        docs = list(docs or [])

        route_label, routing_latency_ms = self._route_query(query)
        rewritten_query, rewrite_latency_ms = self._rewrite_query(query)

        search_queries = self._resolve_search_queries(
            base_query=rewritten_query or query,
            provided_queries=queries,
            expand_queries=expand_queries,
        )

        adaptive_response = self._get_adaptive_retriever().retrieve(
            query=query,
            rewritten_query=rewritten_query,
            queries=search_queries,
            docs=docs,
            user_id=user_id,
            session_id=session_id,
            restrict_to_user_upload=restrict_to_user_upload,
            top_k=top_k,
        )

        retrieved_docs = adaptive_response.documents
        compressed_docs, compression_latency_ms = self._compress_documents(
            query=rewritten_query or query,
            docs=retrieved_docs,
        )

        answer_reflection = self._get_answer_reflection().generate(
            query=query,
            docs=compressed_docs,
            retrieved_chunks=documents_to_chunks(compressed_docs),
            retry_docs=retrieved_docs,
        )

        confidence_start = time.perf_counter()
        confidence_output = self._get_confidence_engine().evaluate(
            adaptive_response.reflection_output,
            answer_reflection,
        )
        confidence_latency_ms = (
            time.perf_counter() - confidence_start
        ) * 1000.0

        result = SelfRAGPipelineResult(
            query=query,
            rewritten_query=rewritten_query,
            route=route_label,
            search_queries=adaptive_response.reflection_output.metadata.get(
                "search_queries",
                search_queries,
            ),
            retrieved_chunks=adaptive_response.reflection_output.retrieval_result.retrieved_chunks,
            compressed_chunks=documents_to_chunks(compressed_docs),
            retrieval=adaptive_response.reflection_output,
            answer_reflection=answer_reflection,
            confidence=confidence_output,
            final_answer=answer_reflection.answer.answer,
            metadata={
                "retrieval_history": adaptive_response.history,
                "restrict_to_user_upload": restrict_to_user_upload,
            },
            metrics=PipelineMetrics(
                total_latency_ms=(
                    time.perf_counter() - total_start
                ) * 1000.0,
                rewrite_latency_ms=rewrite_latency_ms,
                routing_latency_ms=routing_latency_ms,
                compression_latency_ms=compression_latency_ms,
                confidence_latency_ms=confidence_latency_ms,
            ),
        )

        return result

    def _route_query(
        self,
        query: str,
    ) -> tuple[str | None, float]:
        """
        Route the query when a router is available.
        """

        router = self._get_router()
        if router is None:
            return None, 0.0

        start = time.perf_counter()
        try:
            label = str(router(query)).strip() or None
        except Exception as exc:
            logger.warning("Self-RAG router failed: %s", exc)
            label = None

        return label, (time.perf_counter() - start) * 1000.0

    def _rewrite_query(
        self,
        query: str,
    ) -> tuple[str | None, float]:
        """
        Rewrite the query when a rewriter is configured.
        """

        rewriter = self._get_query_rewriter()
        if rewriter is None:
            return None, 0.0

        start = time.perf_counter()
        try:
            rewritten = str(rewriter(query)).strip() or query
        except Exception as exc:
            logger.warning("Self-RAG query rewrite failed: %s", exc)
            rewritten = query

        return rewritten, (time.perf_counter() - start) * 1000.0

    def _resolve_search_queries(
        self,
        *,
        base_query: str,
        provided_queries: list[str] | None,
        expand_queries: bool,
    ) -> list[str]:
        """
        Resolve the final set of retrieval queries.
        """

        candidates = [base_query]
        candidates.extend(provided_queries or [])

        if expand_queries:
            generator = self._get_multi_query_generator()
            if generator is not None:
                try:
                    generated = list(generator(base_query) or [])
                    candidates.extend(generated)
                except Exception as exc:
                    logger.warning("Multi-query expansion failed: %s", exc)

        seen: set[str] = set()
        resolved: list[str] = []
        for candidate in candidates:
            normalized = (candidate or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            resolved.append(normalized)

        return resolved or [base_query]

    def _compress_documents(
        self,
        *,
        query: str,
        docs: list[Document],
    ) -> tuple[list[Document], float]:
        """
        Run contextual compression with a safe fallback.
        """

        compressor = self._get_compressor()
        if compressor is None or not docs:
            return list(docs), 0.0

        start = time.perf_counter()

        try:
            compressed = call_with_supported_kwargs(
                compressor,
                query=query,
                retriever_func=lambda q, _: docs,
                docs=docs,
                documents=docs,
            )
            compressed_docs = list(compressed or [])
        except Exception as exc:
            logger.warning("Compression failed; using retrieved docs: %s", exc)
            compressed_docs = list(docs)

        if len(compressed_docs) < min(3, len(docs)):
            compressed_docs = list(docs[: max(3, min(len(docs), 5))])

        return compressed_docs, (time.perf_counter() - start) * 1000.0

    def _get_adaptive_retriever(self) -> AdaptiveRetriever:
        """
        Lazily construct the adaptive retriever.
        """

        if self._adaptive_retriever is None:
            retrieval_reflection = RetrievalReflectionEngine(
                critic=self._get_retrieval_critic(),
                settings=self._settings,
            )

            self._adaptive_retriever = AdaptiveRetriever(
                hybrid_retriever=self._get_hybrid_retriever(),
                mmr_retriever_factory=self._get_mmr_retriever_factory(),
                reranker=self._get_reranker(),
                reranker_scorer=self._get_reranker_scorer(),
                retrieval_reflection=retrieval_reflection,
                retry_strategy=self._retry_strategy,
                settings=self._settings,
            )

        return self._adaptive_retriever

    def _get_answer_reflection(self) -> AnswerReflectionEngine:
        """
        Lazily construct the answer reflection engine.
        """

        if self._answer_reflection is None:
            self._answer_reflection = AnswerReflectionEngine(
                generator=self._get_generator(),
                critic=self._get_answer_critic(),
                retry_strategy=self._retry_strategy,
                answer_cleaner=self._get_answer_cleaner(),
                settings=self._settings,
            )

        return self._answer_reflection

    def _get_llm(self) -> Any:
        """
        Resolve the LLM used by critics.

        Defaults to the local Ollama critic model rather than the Cerebras
        generation model: critique is a grading task, and reusing the
        rate-limited generation model meant each reflective query burned 4-5
        of its ~5 requests/minute. An explicit llm= passed to the constructor
        still wins. See self_rag.utils.critic_model for the model choice.
        """

        if self._llm is None:
            from self_rag.utils.critic_model import critic_model

            self._llm = critic_model

        return self._llm

    def _get_query_rewriter(self) -> Any | None:
        if self._query_rewriter is None:
            try:
                from pre_retrieval.query_rewritter import rewrite_query

                self._query_rewriter = rewrite_query
            except Exception:
                self._query_rewriter = False

        return None if self._query_rewriter is False else self._query_rewriter

    def _get_router(self) -> Any | None:
        if self._router is None:
            try:
                from router.classifier import classify_query

                self._router = classify_query
            except Exception:
                self._router = False

        return None if self._router is False else self._router

    def _get_multi_query_generator(self) -> Any | None:
        if self._multi_query_generator is None:
            try:
                from pre_retrieval.multi_query import generate_multi_queries

                self._multi_query_generator = generate_multi_queries
            except Exception:
                self._multi_query_generator = False

        return (
            None
            if self._multi_query_generator is False
            else self._multi_query_generator
        )

    def _get_hybrid_retriever(self) -> Any:
        if self._hybrid_retriever is None:
            from retrieval.hybrid import hybrid_retrieve

            self._hybrid_retriever = hybrid_retrieve

        return self._hybrid_retriever

    def _get_mmr_retriever_factory(self) -> Any:
        if self._mmr_retriever_factory is None:
            from retrieval.mmr import get_mmr_retriever

            self._mmr_retriever_factory = get_mmr_retriever

        return self._mmr_retriever_factory

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from retrieval.reranker import rerank_documents

            self._reranker = rerank_documents

        return self._reranker

    def _get_reranker_scorer(self) -> Any | None:
        if self._reranker_scorer is None:
            try:
                from retrieval import reranker as reranker_module

                scorer_model = reranker_module._model

                def score_documents(*, query: str, docs: list[Document]) -> list[float]:
                    pairs = [(query, doc.page_content[:800]) for doc in docs]
                    scores = scorer_model.predict(
                        pairs,
                        batch_size=32,
                        show_progress_bar=False,
                    )
                    return [normalize_score(score) for score in scores]

                self._reranker_scorer = score_documents
            except Exception:
                self._reranker_scorer = False

        return (
            None
            if self._reranker_scorer is False
            else self._reranker_scorer
        )

    def _get_compressor(self) -> Any | None:
        if self._compressor is None:
            try:
                from post_retrieval.contextual_compression import (
                    compress_documents,
                )

                self._compressor = compress_documents
            except Exception:
                self._compressor = False

        return None if self._compressor is False else self._compressor

    def _get_generator(self) -> Any:
        if self._generator is None:
            from generation.generate import generate_answer

            self._generator = generate_answer

        return self._generator

    def _get_answer_cleaner(self) -> Any | None:
        if self._answer_cleaner is None:
            try:
                from generation.structured_output import clean_output

                self._answer_cleaner = clean_output
            except Exception:
                self._answer_cleaner = False

        return (
            None
            if self._answer_cleaner is False
            else self._answer_cleaner
        )

    def _get_retrieval_critic(self) -> RetrievalCritic:
        if self._retrieval_critic is None:
            self._retrieval_critic = RetrievalCritic(
                self._get_llm(),
                temperature=self._settings.REFLECTION_TEMPERATURE,
                max_tokens=self._settings.REFLECTION_MAX_TOKENS,
            )

        return self._retrieval_critic

    def _get_answer_critic(self) -> AnswerCritic:
        if self._answer_critic is None:
            self._answer_critic = AnswerCritic(
                self._get_llm(),
                temperature=self._settings.REFLECTION_TEMPERATURE,
                max_tokens=self._settings.REFLECTION_MAX_TOKENS,
            )

        return self._answer_critic

    def _get_confidence_engine(self) -> ConfidenceEngine:
        if self._confidence_engine is None:
            self._confidence_engine = ConfidenceEngine(
                settings=self._settings
            )

        return self._confidence_engine


def build_empty_result(
    query: str,
    *,
    rewritten_query: str | None = None,
    route: str | None = None,
    final_answer: str = "",
) -> SelfRAGPipelineResult:
    """
    Construct a safe empty result for callers that need a placeholder.
    """

    answer_text = final_answer or "No answer generated."

    retrieval_output = RetrievalReflectionOutput(
        success=False,
        retrieval_result=AdaptiveRetrievalResult(
            iterations=1,
            final_top_k=1,
            retrieval_decision=RetrievalDecision(
                sufficient_context=False,
                retrieval_confidence=0.0,
                suggested_top_k=1,
                reasoning="No retrieval was executed.",
                reflection_metadata={},
            ),
            retrieved_chunks=[],
        ),
        metrics=RetrievalMetrics(),
        metadata={},
    )

    answer_output = AnswerReflectionOutput(
        answer=GeneratedAnswer(answer=answer_text, citations=[], metadata={}),
        evaluation=AnswerEvaluation(
            grounded=False,
            faithful=False,
            hallucination_detected=False,
            confidence=0.0,
            retry_required=False,
            reasoning="No answer generation was executed.",
            missing_information=[],
        ),
        metrics=AnswerMetrics(),
        retries=0,
    )

    confidence_output = ConfidenceEngineOutput(
        confidence=ConfidenceResult(
            overall_score=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            passed_threshold=False,
            breakdown=ConfidenceBreakdown(
                retrieval=ConfidenceComponent(
                    name="retrieval",
                    score=0.0,
                    weight=0.0,
                    reasoning="No retrieval executed.",
                ),
                reranker=ConfidenceComponent(
                    name="reranker",
                    score=0.0,
                    weight=0.0,
                    reasoning="No reranking executed.",
                ),
                answer=ConfidenceComponent(
                    name="answer",
                    score=0.0,
                    weight=0.0,
                    reasoning="No answer generation executed.",
                ),
                overall=0.0,
            ),
            reasoning="No pipeline execution occurred.",
        ),
        metrics=ConfidenceMetrics(),
    )

    return SelfRAGPipelineResult(
        query=query,
        rewritten_query=rewritten_query,
        route=route,
        search_queries=[],
        retrieved_chunks=[],
        compressed_chunks=[],
        retrieval=retrieval_output,
        answer_reflection=answer_output,
        confidence=confidence_output,
        final_answer=answer_text,
        metadata={},
        metrics=PipelineMetrics(),
    )
