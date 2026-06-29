"""
self_rag.schemas.answer
=======================

Schemas for answer generation, evaluation, and reflection.

These models define the contract between:

- Generator
- Answer Critic
- Answer Reflection
- Confidence Engine

This module contains NO business logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """
    Citation associated with an answer.
    """

    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(
        ...,
        description="Document identifier.",
    )

    chunk_id: str = Field(
        ...,
        description="Chunk identifier.",
    )

    source: str = Field(
        ...,
        description="Human-readable source name.",
    )

    page: int | None = Field(
        default=None,
        ge=1,
        description="Page number if available.",
    )

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence.",
    )


class GeneratedAnswer(BaseModel):
    """
    Raw answer produced by the LLM.
    """

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(
        ...,
        min_length=1,
    )

    citations: list[Citation] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class AnswerEvaluation(BaseModel):
    """
    Evaluation produced by the Answer Critic.
    """

    model_config = ConfigDict(extra="ignore")

    grounded: bool = Field(
        ...,
        description="Whether the answer is supported by retrieved context.",
    )

    faithful: bool = Field(
        ...,
        description="Whether the answer faithfully represents the retrieved evidence.",
    )

    hallucination_detected: bool = Field(
        default=False,
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    retry_required: bool = Field(
        default=False,
    )

    reasoning: str = Field(
        ...,
    )

    missing_information: list[str] = Field(
        default_factory=list,
    )


class AnswerMetrics(BaseModel):
    """
    Runtime metrics collected during answer generation.
    """

    model_config = ConfigDict(extra="ignore")

    generation_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    reflection_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
    )

    total_tokens: int = Field(
        default=0,
        ge=0,
    )


class AnswerReflectionOutput(BaseModel):
    """
    Final output returned by the Answer Reflection pipeline.
    """

    model_config = ConfigDict(extra="ignore")

    answer: GeneratedAnswer

    evaluation: AnswerEvaluation

    metrics: AnswerMetrics

    retries: int = Field(
        default=0,
        ge=0,
    )