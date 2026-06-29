"""
self_rag.schemas.retrieval
==========================

Pydantic schemas used by the Self-RAG retrieval reflection pipeline.

These models define the contract between:

- Retriever
- Retrieval Critic
- Adaptive Retriever
- Reflection Pipeline

Business logic MUST NOT be implemented here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """
    Represents a retrieved document chunk.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=False,
        validate_assignment=True,
    )

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier.",
    )

    document_id: str = Field(
        ...,
        description="Parent document identifier.",
    )

    text: str = Field(
        ...,
        description="Chunk text.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata.",
    )

    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Retriever similarity score.",
    )

    reranker_score: float = Field(
        default=0.0,
        ge=0.0,
        description="CrossEncoder reranker score.",
    )

    rank: int = Field(
        default=0,
        ge=0,
        description="Final ranking position.",
    )


class RetrievalRequest(BaseModel):
    """
    Input to Retrieval Critic.
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        description="User query.",
    )

    rewritten_query: str | None = Field(
        default=None,
        description="Optional rewritten query.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    iteration: int = Field(
        default=1,
        ge=1,
    )

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
    )


class RetrievalDecision(BaseModel):
    """
    Decision returned by Retrieval Critic.
    """

    model_config = ConfigDict(extra="ignore")

    sufficient_context: bool = Field(
        ...,
        description="Whether retrieved context is sufficient.",
    )

    retrieval_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    suggested_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    reasoning: str = Field(
        ...,
        description="Reasoning behind the decision.",
    )

    reflection_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class AdaptiveRetrievalResult(BaseModel):
    """
    Output of Adaptive Retriever.
    """

    model_config = ConfigDict(extra="ignore")

    iterations: int = Field(
        default=1,
        ge=1,
    )

    final_top_k: int = Field(
        default=5,
        ge=1,
    )

    retrieval_decision: RetrievalDecision

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
    )


class RetrievalMetrics(BaseModel):
    """
    Runtime metrics collected during retrieval.
    """

    model_config = ConfigDict(extra="ignore")

    retrieval_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    reranker_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    reflection_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    retrieved_document_count: int = Field(
        default=0,
        ge=0,
    )

    average_reranker_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class RetrievalReflectionOutput(BaseModel):
    """
    Final output returned by the Retrieval Reflection stage.
    """

    model_config = ConfigDict(extra="ignore")

    success: bool = Field(
        default=True,
    )

    retrieval_result: AdaptiveRetrievalResult

    metrics: RetrievalMetrics

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )