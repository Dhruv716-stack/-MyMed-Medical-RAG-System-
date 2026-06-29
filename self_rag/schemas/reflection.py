"""
self_rag.schemas.reflection
===========================

Generic reflection schemas shared across the Self-RAG pipeline.

These schemas represent the evaluation produced by reflection
components (retrieval critic, answer critic, future coverage
critic, etc.).

No business logic belongs here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReflectionStatus(str, Enum):
    """
    Overall reflection decision.
    """

    PASS = "pass"
    RETRY = "retry"
    FAIL = "fail"


class ReflectionType(str, Enum):
    """
    Reflection stage.
    """

    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    COVERAGE = "coverage"


class ReflectionReason(BaseModel):
    """
    A single explanation produced during reflection.
    """

    model_config = ConfigDict(extra="ignore")

    category: str = Field(
        ...,
        description="Reason category."
    )

    message: str = Field(
        ...,
        description="Human-readable explanation."
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence for this reason."
    )


class ReflectionRequest(BaseModel):
    """
    Generic reflection request.
    """

    model_config = ConfigDict(extra="ignore")

    reflection_type: ReflectionType

    query: str

    context: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    """
    Generic reflection response.

    This schema is reused by all critics.
    """

    model_config = ConfigDict(extra="ignore")

    status: ReflectionStatus

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    passed: bool

    retry_required: bool

    reasoning: str

    reasons: list[ReflectionReason] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ReflectionMetrics(BaseModel):
    """
    Metrics collected during reflection.
    """

    model_config = ConfigDict(extra="ignore")

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    llm_tokens: int = Field(
        default=0,
        ge=0,
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
    )


class ReflectionOutput(BaseModel):
    """
    Final output returned by any reflection stage.
    """

    model_config = ConfigDict(extra="ignore")

    result: ReflectionResult

    metrics: ReflectionMetrics

    timestamp: str | None = None