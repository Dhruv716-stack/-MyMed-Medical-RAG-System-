"""
self_rag.critics.answer_critic
==============================

Production-grade Answer Critic for validating whether generated
responses remain grounded in retrieved medical evidence.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from self_rag.prompts.answer_prompt import build_answer_prompt
from self_rag.schemas.answer import AnswerEvaluation
from self_rag.schemas.retrieval import RetrievedChunk
from self_rag.utils.llm import (
    LLMInvocationError,
    ainvoke_llm,
    extract_json_payload,
    invoke_llm,
)

logger = logging.getLogger(__name__)


class AnswerCriticError(Exception):
    """Base exception for the Answer Critic."""


class AnswerCriticResponseError(AnswerCriticError):
    """Raised when the LLM returns invalid JSON."""


class AnswerCriticInvocationError(AnswerCriticError):
    """Raised when the underlying LLM call fails."""


class AnswerCriticRequest(BaseModel):
    """
    Structured request passed to the Answer Critic.
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1)
    generated_answer: str = Field(..., min_length=1)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class AnswerCritic:
    """
    Evaluate answer faithfulness and grounding.
    """

    def __init__(
        self,
        llm: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def evaluate(
        self,
        request: AnswerCriticRequest,
    ) -> AnswerEvaluation:
        """
        Async answer evaluation.
        """

        messages = self._build_messages(request)
        raw_response = await self._call_llm_async(messages)
        payload = self._parse_response(raw_response)
        return self._create_evaluation(payload)

    def evaluate_sync(
        self,
        request: AnswerCriticRequest,
    ) -> AnswerEvaluation:
        """
        Sync answer evaluation for direct pipeline usage.
        """

        messages = self._build_messages(request)
        raw_response = self._call_llm(messages)
        payload = self._parse_response(raw_response)
        return self._create_evaluation(payload)

    def _build_messages(
        self,
        request: AnswerCriticRequest,
    ) -> list[dict[str, str]]:
        """
        Build the Answer Critic prompt.
        """

        chunk_texts = [chunk.text for chunk in request.retrieved_chunks]

        return build_answer_prompt(
            query=request.query,
            retrieved_chunks=chunk_texts,
            generated_answer=request.generated_answer,
        )

    def _call_llm(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Sync LLM invocation.
        """

        try:
            return invoke_llm(
                self._llm,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LLMInvocationError as exc:
            logger.exception("Answer Critic LLM invocation failed.")
            raise AnswerCriticInvocationError(
                "Unable to invoke Answer Critic."
            ) from exc

    async def _call_llm_async(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Async LLM invocation.
        """

        try:
            return await ainvoke_llm(
                self._llm,
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LLMInvocationError as exc:
            logger.exception("Answer Critic async invocation failed.")
            raise AnswerCriticInvocationError(
                "Unable to invoke Answer Critic."
            ) from exc

    def _parse_response(
        self,
        raw_response: str,
    ) -> dict[str, Any]:
        """
        Parse and validate JSON payload.
        """

        try:
            payload = extract_json_payload(raw_response)
        except Exception as exc:
            logger.exception("Answer Critic returned invalid JSON.")
            raise AnswerCriticResponseError(
                "Invalid JSON returned by Answer Critic."
            ) from exc

        return payload

    def _create_evaluation(
        self,
        payload: dict[str, Any],
    ) -> AnswerEvaluation:
        """
        Validate the answer evaluation payload.
        """

        normalized_payload = dict(payload)
        normalized_payload.setdefault("missing_information", [])

        try:
            return AnswerEvaluation.model_validate(normalized_payload)
        except Exception as exc:
            logger.exception("AnswerEvaluation validation failed.")
            raise AnswerCriticResponseError(
                "Invalid Answer Critic response."
            ) from exc
