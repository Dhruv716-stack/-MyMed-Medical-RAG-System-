"""
self_rag.critics.confidence
===========================

Confidence engine for combining retrieval quality, reranker strength,
and answer faithfulness into a final confidence estimate.
"""

from __future__ import annotations

import logging
import time

from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.schemas.answer import AnswerReflectionOutput
from self_rag.schemas.confidence import (
    ConfidenceBreakdown,
    ConfidenceComponent,
    ConfidenceEngineOutput,
    ConfidenceLevel,
    ConfidenceMetrics,
    ConfidenceResult,
    ConfidenceThresholds,
)
from self_rag.schemas.retrieval import RetrievalReflectionOutput

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """
    Combine Self-RAG stage signals into one final confidence score.
    """

    def __init__(
        self,
        *,
        settings: SelfRAGSettings | None = None,
        thresholds: ConfidenceThresholds | None = None,
        minimum_passing_score: float | None = None,
    ) -> None:
        self._settings = settings or get_self_rag_settings()
        self._thresholds = thresholds or ConfidenceThresholds()
        self._minimum_passing_score = (
            minimum_passing_score
            if minimum_passing_score is not None
            else self._settings.ANSWER_CONFIDENCE_THRESHOLD
        )

    def evaluate(
        self,
        retrieval_output: RetrievalReflectionOutput,
        answer_output: AnswerReflectionOutput,
    ) -> ConfidenceEngineOutput:
        """
        Produce a structured confidence estimate.
        """

        start = time.perf_counter()

        retrieval_decision = retrieval_output.retrieval_result.retrieval_decision
        answer_evaluation = answer_output.evaluation

        retrieval_component = ConfidenceComponent(
            name="retrieval",
            score=retrieval_decision.retrieval_confidence,
            weight=self._settings.RETRIEVAL_WEIGHT,
            reasoning=retrieval_decision.reasoning,
        )

        reranker_component = ConfidenceComponent(
            name="reranker",
            score=retrieval_output.metrics.average_reranker_score,
            weight=self._settings.RERANKER_WEIGHT,
            reasoning=(
                f"Average normalized reranker score across "
                f"{retrieval_output.metrics.retrieved_document_count} chunks."
            ),
        )

        answer_component = ConfidenceComponent(
            name="answer",
            score=answer_evaluation.confidence,
            weight=self._settings.ANSWER_WEIGHT,
            reasoning=answer_evaluation.reasoning,
        )

        overall_score = (
            retrieval_component.score * retrieval_component.weight
            + reranker_component.score * reranker_component.weight
            + answer_component.score * answer_component.weight
        )

        confidence_level = self._resolve_confidence_level(overall_score)

        passed_threshold = (
            overall_score >= self._minimum_passing_score
            and retrieval_decision.sufficient_context
            and answer_evaluation.grounded
            and answer_evaluation.faithful
            and not answer_evaluation.hallucination_detected
        )

        reasoning = (
            f"Retrieval={retrieval_component.score:.2f}, "
            f"Reranker={reranker_component.score:.2f}, "
            f"Answer={answer_component.score:.2f}. "
            f"Overall={overall_score:.2f} ({confidence_level.value})."
        )

        result = ConfidenceResult(
            overall_score=overall_score,
            confidence_level=confidence_level,
            passed_threshold=passed_threshold,
            breakdown=ConfidenceBreakdown(
                retrieval=retrieval_component,
                reranker=reranker_component,
                answer=answer_component,
                overall=overall_score,
            ),
            reasoning=reasoning,
        )

        metrics = ConfidenceMetrics(
            latency_ms=(time.perf_counter() - start) * 1000.0,
            retrieval_score=retrieval_component.score,
            reranker_score=reranker_component.score,
            answer_score=answer_component.score,
        )

        logger.info(
            "Confidence computed (overall=%.3f, passed=%s)",
            overall_score,
            passed_threshold,
        )

        return ConfidenceEngineOutput(
            confidence=result,
            metrics=metrics,
        )

    def _resolve_confidence_level(
        self,
        score: float,
    ) -> ConfidenceLevel:
        """
        Map a numeric score to a human-readable confidence level.
        """

        if score >= self._thresholds.very_high:
            return ConfidenceLevel.VERY_HIGH
        if score >= self._thresholds.high:
            return ConfidenceLevel.HIGH
        if score >= self._thresholds.medium:
            return ConfidenceLevel.MEDIUM
        if score >= self._thresholds.low:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW
