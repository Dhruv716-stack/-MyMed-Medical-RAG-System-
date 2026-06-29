"""
self_rag.config.settings
========================

Configuration settings for the Self-RAG subsystem.

This module centralizes all configurable parameters used by the
reflection pipeline. Business logic must NEVER be implemented here.

Author:
    MyMED Development Team

Python:
    3.12+
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SelfRAGSettings(BaseSettings):
    """
    Configuration for the Self-RAG pipeline.

    Values can be overridden using environment variables with the prefix:

        SELF_RAG_

    Example:

        SELF_RAG_MAX_RETRIEVAL_ITERATIONS=3
    """

    model_config = SettingsConfigDict(
        env_prefix="SELF_RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    # ==========================================================
    # Retrieval Reflection
    # ==========================================================

    ENABLE_RETRIEVAL_REFLECTION: bool = Field(
        default=True,
        description="Enable retrieval reflection stage.",
    )

    MAX_RETRIEVAL_ITERATIONS: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum retrieval retries.",
    )

    INITIAL_TOP_K: int = Field(
        default=6,
        ge=1,
        le=50,
    )

    TOP_K_INCREMENT: int = Field(
        default=3,
        ge=1,
        le=20,
    )

    RETRIEVAL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )

    MIN_RERANKER_SCORE: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )

    # ==========================================================
    # Answer Reflection
    # ==========================================================

    ENABLE_ANSWER_REFLECTION: bool = Field(
        default=True,
    )

    MAX_GENERATION_RETRIES: int = Field(
        default=1,
        ge=0,
        le=2,
    )

    ANSWER_CONFIDENCE_THRESHOLD: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    # ==========================================================
    # LLM Parameters
    # ==========================================================

    REFLECTION_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    REFLECTION_MAX_TOKENS: int = Field(
        default=512,
        ge=64,
        le=4096,
    )

    # ==========================================================
    # Logging
    # ==========================================================

    ENABLE_DEBUG_LOGGING: bool = Field(
        default=False,
    )

    ENABLE_LATENCY_LOGGING: bool = Field(
        default=True,
    )

    # ==========================================================
    # Confidence Engine
    # ==========================================================

    RETRIEVAL_WEIGHT: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
    )

    ANSWER_WEIGHT: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )

    RERANKER_WEIGHT: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )

    # ==========================================================
    # Validation
    # ==========================================================

    def validate_weights(self) -> None:
        """
        Ensure confidence weights sum to approximately 1.0.
        """
        total = (
            self.RETRIEVAL_WEIGHT
            + self.ANSWER_WEIGHT
            + self.RERANKER_WEIGHT
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Confidence weights must sum to 1.0 "
                f"(current={total:.3f})"
            )


@lru_cache
def get_self_rag_settings() -> SelfRAGSettings:
    """
    Cached settings singleton.

    Returns
    -------
    SelfRAGSettings
        Loaded configuration instance.
    """
    settings = SelfRAGSettings()
    settings.validate_weights()
    return settings