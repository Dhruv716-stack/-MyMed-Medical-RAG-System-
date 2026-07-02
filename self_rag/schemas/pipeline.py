"""
self_rag.schemas.pipeline
=========================

Structured output models for the end-to-end Self-RAG pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .answer import AnswerReflectionOutput
from .confidence import ConfidenceEngineOutput
from .retrieval import RetrievedChunk, RetrievalReflectionOutput


class PipelineMetrics(BaseModel):
    """
    Stage-level runtime metrics for Self-RAG.
    """

    model_config = ConfigDict(extra="ignore")

    total_latency_ms: float = Field(default=0.0, ge=0.0)
    rewrite_latency_ms: float = Field(default=0.0, ge=0.0)
    routing_latency_ms: float = Field(default=0.0, ge=0.0)
    compression_latency_ms: float = Field(default=0.0, ge=0.0)
    confidence_latency_ms: float = Field(default=0.0, ge=0.0)


class SelfRAGPipelineResult(BaseModel):
    """
    End-to-end result returned by ``SelfRAGPipeline``.
    """

    model_config = ConfigDict(extra="ignore")

    query: str
    rewritten_query: str | None = None
    route: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    compressed_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval: RetrievalReflectionOutput
    answer_reflection: AnswerReflectionOutput
    confidence: ConfidenceEngineOutput
    final_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
