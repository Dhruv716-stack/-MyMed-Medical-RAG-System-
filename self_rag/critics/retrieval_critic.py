"""
self_rag.critics.retrieval_critic
=================================

Production-grade Retrieval Critic for the MyMED Self-RAG workflow.

The Retrieval Critic evaluates the quality of retrieved
medical evidence and determines whether the retrieved
context is sufficient for answer generation.

Responsibilities
----------------
- Evaluate retrieval quality
- Estimate retrieval confidence
- Detect insufficient context
- Recommend Top-K adjustments
- Return structured RetrievalDecision

The critic NEVER:
- Performs retrieval
- Retries retrieval
- Generates answers
- Calls web search
"""

from __future__ import annotations

import logging
from typing import Any

from self_rag.prompts.retrieval_prompt import build_retrieval_prompt
from self_rag.schemas.retrieval import RetrievalDecision, RetrievalRequest
from self_rag.utils.llm import (
    LLMInvocationError,
    ainvoke_llm,
    extract_json_payload,
    invoke_llm,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Exceptions
# ==========================================================


class RetrievalCriticError(Exception):
    """Base exception for Retrieval Critic."""


class CriticResponseError(RetrievalCriticError):
    """Raised when the LLM returns an invalid response."""


class CriticInvocationError(RetrievalCriticError):
    """Raised when the underlying LLM invocation fails."""


# ==========================================================
# Retrieval Critic
# ==========================================================


class RetrievalCritic:
    """
    Evaluates retrieval quality.

    Notes
    -----
    This class contains no retrieval logic.

    It only evaluates retrieval results.
    """

    def __init__(
        self,
        llm,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """
        Parameters
        ----------
        llm
            Existing MyMED LLM provider.

        temperature
            Reflection temperature.

        max_tokens
            Reflection token limit.
        """

        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ======================================================
    # Public API
    # ======================================================

    async def evaluate(
        self,
        request: RetrievalRequest,
    ) -> RetrievalDecision:
        """
        Evaluate retrieval quality.

        Parameters
        ----------
        request
            Retrieval request.

        Returns
        -------
        RetrievalDecision

        Raises
        ------
        RetrievalCriticError
        """

        logger.info(
            "Starting Retrieval Critic "
            "(iteration=%d)",
            request.iteration,
        )

        messages = self._build_messages(request)

        raw_response = await self._call_llm_async(messages)

        payload = self._parse_response(raw_response)

        decision = self._create_decision(payload, request)

        logger.info(
            "Retrieval evaluation complete "
            "(confidence=%.3f, sufficient=%s)",
            decision.retrieval_confidence,
            decision.sufficient_context,
        )

        return decision

    def evaluate_sync(
        self,
        request: RetrievalRequest,
    ) -> RetrievalDecision:
        """
        Synchronous evaluation variant for the existing pipeline.
        """

        logger.info(
            "Starting Retrieval Critic "
            "(iteration=%d)",
            request.iteration,
        )

        messages = self._build_messages(request)

        raw_response = self._call_llm(messages)

        payload = self._parse_response(raw_response)

        decision = self._create_decision(payload, request)

        logger.info(
            "Retrieval evaluation complete "
            "(confidence=%.3f, sufficient=%s)",
            decision.retrieval_confidence,
            decision.sufficient_context,
        )

        return decision

    # ======================================================
    # Prompt Builder
    # ======================================================

    def _build_messages(
        self,
        request: RetrievalRequest,
    ) -> list[dict[str, str]]:
        """
        Build Retrieval Critic prompt.
        """

        chunk_texts = [
            chunk.text
            for chunk in request.retrieved_chunks
        ]

        reranker_scores = [
            chunk.reranker_score
            for chunk in request.retrieved_chunks
        ]

        return build_retrieval_prompt(
            query=request.query,
            retrieved_chunks=chunk_texts,
            reranker_scores=reranker_scores,
        )

    # ======================================================
    # LLM Invocation
    # ======================================================

    def _call_llm(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Invoke the underlying LLM.

        Parameters
        ----------
        messages
            Prompt messages.

        Returns
        -------
        str
            Raw LLM response.

        Raises
        ------
        CriticInvocationError
        """

        try:
            return invoke_llm(
                self._llm,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LLMInvocationError as exc:
            logger.exception("Retrieval Critic LLM invocation failed.")
            raise CriticInvocationError(
                "Unable to invoke Retrieval Critic."
            ) from exc

    async def _call_llm_async(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Invoke the underlying LLM asynchronously.
        """

        try:
            return await ainvoke_llm(
                self._llm,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LLMInvocationError as exc:
            logger.exception("Retrieval Critic async invocation failed.")
            raise CriticInvocationError(
                "Unable to invoke Retrieval Critic."
            ) from exc

    # ======================================================
    # Response Parsing
    # ======================================================

    def _parse_response(
        self,
        raw_response: str,
    ) -> dict[str, Any]:
        """
        Parse Retrieval Critic JSON response.

        Parameters
        ----------
        raw_response
            Raw response returned by the LLM.

        Returns
        -------
        dict[str, Any]

        Raises
        ------
        CriticResponseError
        """

        try:
            payload = extract_json_payload(raw_response)
        except Exception as exc:
            logger.exception("Retrieval Critic returned invalid JSON.")
            raise CriticResponseError(
                "Invalid JSON returned by Retrieval Critic."
            ) from exc

        return payload

    # ======================================================
    # Decision Builder
    # ======================================================

    def _create_decision(
        self,
        payload: dict[str, Any],
        request: RetrievalRequest,
    ) -> RetrievalDecision:
        """
        Convert the validated JSON payload into a
        RetrievalDecision.

        Parameters
        ----------
        payload
            Parsed JSON returned by the LLM.

        Returns
        -------
        RetrievalDecision

        Raises
        ------
        CriticResponseError
            If payload validation fails.
        """

        normalized_payload = dict(payload)
        if (
            "recommended_top_k" in normalized_payload
            and "suggested_top_k" not in normalized_payload
        ):
            normalized_payload["suggested_top_k"] = normalized_payload.pop(
                "recommended_top_k"
            )

        normalized_payload.setdefault(
            "suggested_top_k",
            request.top_k,
        )
        normalized_payload.setdefault(
            "reflection_metadata",
            {},
        )

        try:
            decision = RetrievalDecision.model_validate(normalized_payload)
        except Exception as exc:
            logger.exception("RetrievalDecision validation failed.")
            raise CriticResponseError(
                "Invalid Retrieval Critic response."
            ) from exc
        return decision
