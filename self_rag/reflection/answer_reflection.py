"""
self_rag.reflection.answer_reflection
=====================================

Answer reflection engine that evaluates generated responses and can
retry generation once with broader context or critique guidance.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.documents import Document

from self_rag.adaptive.retry_strategy import RetryStrategy
from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.critics.answer_critic import (
    AnswerCritic,
    AnswerCriticError,
    AnswerCriticRequest,
)
from self_rag.schemas.answer import (
    AnswerEvaluation,
    AnswerMetrics,
    AnswerReflectionOutput,
    GeneratedAnswer,
)
from self_rag.schemas.retrieval import RetrievedChunk
from self_rag.utils.compat import call_with_supported_kwargs
from self_rag.utils.documents import build_citations, documents_to_chunks
from self_rag.utils.llm import normalize_llm_response

logger = logging.getLogger(__name__)

_INLINE_CITATION_PATTERN = re.compile(r"\[Source:\s*", re.IGNORECASE)


class AnswerReflectionEngine:
    """
    Generate an answer, reflect on it, and optionally retry once.
    """

    def __init__(
        self,
        *,
        generator: Any,
        critic: AnswerCritic | None = None,
        retry_strategy: RetryStrategy | None = None,
        answer_cleaner: Any | None = None,
        settings: SelfRAGSettings | None = None,
    ) -> None:
        self._generator = generator
        self._critic = critic
        self._answer_cleaner = answer_cleaner
        self._settings = settings or get_self_rag_settings()
        self._retry_strategy = retry_strategy or RetryStrategy(
            settings=self._settings
        )

    def generate(
        self,
        query: str,
        docs: list[Document],
        *,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        retry_docs: list[Document] | None = None,
    ) -> AnswerReflectionOutput:
        """
        Generate an answer and validate it with reflection.
        """

        retries_used = 0
        total_generation_latency_ms = 0.0
        total_reflection_latency_ms = 0.0
        retry_guidance: str | None = None
        previous_answer: str | None = None
        evaluation_source = "heuristic"

        base_docs = list(docs or [])
        broader_docs = list(retry_docs or docs or [])
        initial_chunk_context = (
            retrieved_chunks or documents_to_chunks(base_docs)
        )

        final_answer: GeneratedAnswer | None = None
        final_evaluation: AnswerEvaluation | None = None

        while True:
            current_docs = base_docs if retries_used == 0 else broader_docs

            raw_answer, generation_latency_ms = self._generate_answer(
                query=query,
                docs=current_docs,
                retry_guidance=retry_guidance,
                previous_answer=previous_answer,
            )
            total_generation_latency_ms += generation_latency_ms

            cleaned_answer = self._clean_answer(raw_answer)
            current_chunks = documents_to_chunks(current_docs)
            answer_payload = GeneratedAnswer(
                answer=cleaned_answer,
                citations=build_citations(cleaned_answer, current_chunks),
                metadata={
                    "attempt": retries_used + 1,
                    "used_retry_docs": retries_used > 0,
                },
            )

            reflection_start = time.perf_counter()
            evaluation_chunks = (
                initial_chunk_context
                if retries_used == 0 and initial_chunk_context
                else current_chunks
            )
            evaluation, evaluation_source = self._evaluate_answer(
                query=query,
                answer=cleaned_answer,
                chunks=evaluation_chunks,
            )
            total_reflection_latency_ms += (
                time.perf_counter() - reflection_start
            ) * 1000.0

            final_answer = answer_payload
            final_evaluation = evaluation

            if not self._retry_strategy.should_retry_answer(
                evaluation,
                retries_used,
            ):
                break

            retries_used += 1
            previous_answer = cleaned_answer
            retry_guidance = self._build_retry_guidance(evaluation)

        metrics = AnswerMetrics(
            generation_latency_ms=total_generation_latency_ms,
            reflection_latency_ms=total_reflection_latency_ms,
            total_tokens=0,
        )

        if final_answer is None or final_evaluation is None:
            fallback_text = self._fallback_answer()
            final_answer = GeneratedAnswer(
                answer=fallback_text,
                citations=[],
                metadata={"attempt": 0},
            )
            final_evaluation = self._heuristic_evaluation(
                answer=fallback_text,
                chunks=[],
            )

        final_answer.metadata["evaluation_source"] = evaluation_source

        return AnswerReflectionOutput(
            answer=final_answer,
            evaluation=final_evaluation,
            metrics=metrics,
            retries=retries_used,
        )

    def _generate_answer(
        self,
        *,
        query: str,
        docs: list[Document],
        retry_guidance: str | None,
        previous_answer: str | None,
    ) -> tuple[str, float]:
        """
        Invoke the answer generator with optional retry guidance.
        """

        start = time.perf_counter()

        try:
            response = call_with_supported_kwargs(
                self._generator,
                query=query,
                docs=docs,
                documents=docs,
                retry_guidance=retry_guidance,
                feedback=retry_guidance,
                critique=retry_guidance,
                previous_answer=previous_answer,
            )
            answer_text = normalize_llm_response(response)
        except Exception as exc:
            logger.warning(
                "Answer generation failed; returning fallback answer: %s",
                exc,
            )
            answer_text = self._fallback_answer()

        return answer_text, (time.perf_counter() - start) * 1000.0

    def _clean_answer(
        self,
        answer: str,
    ) -> str:
        """
        Normalize answer text with the configured cleaner when present.
        """

        cleaned = answer.strip()

        if self._answer_cleaner is None:
            return cleaned

        try:
            return str(self._answer_cleaner(cleaned)).strip() or cleaned
        except Exception:
            return cleaned

    def _evaluate_answer(
        self,
        *,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[AnswerEvaluation, str]:
        """
        Evaluate an answer with the critic or a heuristic fallback.
        """

        if (
            self._settings.ENABLE_ANSWER_REFLECTION
            and self._critic is not None
        ):
            try:
                request = AnswerCriticRequest(
                    query=query,
                    generated_answer=answer,
                    retrieved_chunks=chunks,
                )
                evaluation = self._critic.evaluate_sync(request)
                return evaluation, "critic"
            except AnswerCriticError as exc:
                logger.warning(
                    "Answer critic failed; using heuristic evaluation: %s",
                    exc,
                )

        return self._heuristic_evaluation(answer=answer, chunks=chunks), "heuristic"

    def _heuristic_evaluation(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerEvaluation:
        """
        Heuristic evaluation used when the critic is unavailable.
        """

        normalized_answer = answer.strip().lower()
        has_context = bool(chunks)
        has_citations = bool(_INLINE_CITATION_PATTERN.search(answer))

        fallback_markers = (
            "not contain enough information",
            "not enough reliable information",
            "could not find enough reliable information",
        )
        unsafe_markers = (
            "as an ai",
            "based on my knowledge",
            "generally speaking",
        )

        is_fallback = any(
            marker in normalized_answer for marker in fallback_markers
        )
        hallucination_detected = any(
            marker in normalized_answer for marker in unsafe_markers
        )

        grounded = has_context or is_fallback
        faithful = grounded and not hallucination_detected

        confidence = 0.30
        if faithful and is_fallback:
            confidence = 0.55 if has_context else 0.40
        elif faithful:
            confidence = 0.60
            if has_citations:
                confidence += 0.10
            confidence += min(len(chunks), 4) * 0.05
            confidence += min(len(answer) / 800.0, 1.0) * 0.10
            confidence = min(confidence, 0.90)

        missing_information = []
        if not has_context:
            missing_information.append("No retrieved context available.")
        if not has_citations and faithful and not is_fallback:
            missing_information.append("Inline source citations are missing.")

        return AnswerEvaluation(
            grounded=grounded,
            faithful=faithful,
            hallucination_detected=hallucination_detected,
            confidence=confidence,
            retry_required=(
                confidence < self._settings.ANSWER_CONFIDENCE_THRESHOLD
                or not faithful
            ),
            reasoning=(
                "Heuristic evaluation based on context availability, "
                "citations, and answer safety markers."
            ),
            missing_information=missing_information,
        )

    def _build_retry_guidance(
        self,
        evaluation: AnswerEvaluation,
    ) -> str:
        """
        Convert evaluation findings into concise retry guidance.
        """

        details = [evaluation.reasoning.strip()]
        if evaluation.missing_information:
            details.append(
                "Address missing information: "
                + "; ".join(evaluation.missing_information)
            )

        if evaluation.hallucination_detected:
            details.append(
                "Remove unsupported claims and ground every statement in the context."
            )

        return " ".join(part for part in details if part)

    def _fallback_answer(self) -> str:
        """
        Safe fallback when answer generation fails.
        """

        return (
            "The provided documents do not contain enough information "
            "to answer this question safely."
        )
