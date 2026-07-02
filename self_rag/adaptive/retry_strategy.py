"""
self_rag.adaptive.retry_strategy
================================

Retry policy for retrieval reflection and answer reflection loops.
"""

from __future__ import annotations

from self_rag.config.setting import (
    SelfRAGSettings,
    get_self_rag_settings,
)
from self_rag.schemas.answer import AnswerEvaluation
from self_rag.schemas.retrieval import RetrievalDecision


class RetryStrategy:
    """
    Central policy for deciding when Self-RAG should retry.
    """

    def __init__(
        self,
        *,
        settings: SelfRAGSettings | None = None,
    ) -> None:
        self._settings = settings or get_self_rag_settings()

    def should_retry_retrieval(
        self,
        decision: RetrievalDecision,
        iteration: int,
    ) -> bool:
        """
        Decide whether retrieval should be retried.
        """

        if iteration >= self._settings.MAX_RETRIEVAL_ITERATIONS:
            return False

        if (
            decision.sufficient_context
            and decision.retrieval_confidence
            >= self._settings.RETRIEVAL_CONFIDENCE_THRESHOLD
        ):
            return False

        return True

    def next_top_k(
        self,
        current_top_k: int,
        decision: RetrievalDecision,
    ) -> int:
        """
        Compute the next Top-K for adaptive retrieval.
        """

        suggested = max(current_top_k, decision.suggested_top_k)
        incremented = current_top_k + self._settings.TOP_K_INCREMENT
        return min(50, max(suggested, incremented))

    def should_retry_answer(
        self,
        evaluation: AnswerEvaluation,
        retries_used: int,
    ) -> bool:
        """
        Decide whether answer generation should be retried.
        """

        if retries_used >= self._settings.MAX_GENERATION_RETRIES:
            return False

        passed_quality_gate = (
            evaluation.grounded
            and evaluation.faithful
            and not evaluation.hallucination_detected
            and evaluation.confidence
            >= self._settings.ANSWER_CONFIDENCE_THRESHOLD
        )

        if passed_quality_gate and not evaluation.retry_required:
            return False

        return True
