"""
self_rag.schemas.confidence
===========================

Confidence schemas for the Self-RAG pipeline.

These schemas define the contract for confidence estimation across
retrieval, generation, reflection, and the final response.

Business logic MUST NOT be implemented here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceLevel(str, Enum):
    """
    Human-readable confidence categories.
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ConfidenceComponent(BaseModel):
    """
    Represents one component contributing to
    the final confidence score.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        description="Component name.",
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized confidence score.",
    )

    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Component weight.",
    )

    reasoning: str = Field(
        default="",
        description="Explanation for the score.",
    )


class ConfidenceBreakdown(BaseModel):
    """
    Individual confidence components.
    """

    model_config = ConfigDict(extra="ignore")

    retrieval: ConfidenceComponent

    reranker: ConfidenceComponent

    answer: ConfidenceComponent

    overall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class ConfidenceResult(BaseModel):
    """
    Final confidence estimation returned by
    the Confidence Engine.
    """

    model_config = ConfigDict(extra="ignore")

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    confidence_level: ConfidenceLevel

    passed_threshold: bool

    breakdown: ConfidenceBreakdown

    reasoning: str


class ConfidenceThresholds(BaseModel):
    """
    Configurable confidence thresholds.
    """

    model_config = ConfigDict(extra="ignore")

    very_high: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
    )

    high: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    medium: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
    )

    low: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )


class ConfidenceMetrics(BaseModel):
    """
    Runtime metrics collected by the
    confidence engine.
    """

    model_config = ConfigDict(extra="ignore")

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    reranker_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    answer_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class ConfidenceEngineOutput(BaseModel):
    """
    Output returned by the Confidence Engine.
    """

    model_config = ConfigDict(extra="ignore")

    confidence: ConfidenceResult

    metrics: ConfidenceMetrics