"""
self_rag.utils.critic_model
===========================

The LLM used by the Self-RAG critics (Retrieval Critic + Answer Critic).

Why this is separate from the generation model
----------------------------------------------
The critics previously reused ``generation.retrieve_model.model`` (Cerebras
gpt-oss-120b). That model also serves query classification, answer generation,
general chat and summarisation, so every self-reflective query spent 4-5 calls
against a provider whose free tier allows only ~5 requests/minute — the critics
alone doubled the load and made 429s far more likely.

Critique is a grading task, not a generation task: read the evidence, judge the
answer, emit a small JSON object. That runs fine on a local model, so the
critics now use Ollama. This keeps the rate-limited cloud model for the work
that actually needs it (writing the answer).

Model choice
------------
``llama3.1:8b`` (override with SELF_RAG_CRITIC_MODEL).

Measured on the real critic prompt, judging a correctly grounded answer:

    llama3.1:8b  ~10s warm  -> grounded=True,  confidence=1.0   (correct)
    qwen2.5:3b   ~10s       -> grounded=False, confidence=0.25  (wrong)

Both were equally fast once warm, but the 3B model wrongly reported a
hallucination. A critic that fails good answers is worse than no critic: it
trips the confidence threshold and triggers a needless regeneration, costing
an extra generation call and more latency than it saves.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


CRITIC_MODEL_NAME = os.getenv("SELF_RAG_CRITIC_MODEL", "llama3.1:8b")

CRITIC_BASE_URL = os.getenv(
    "SELF_RAG_CRITIC_BASE_URL",
    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)


# Temperature/max_tokens are bound per call by self_rag.utils.llm from
# SelfRAGSettings, so they are intentionally not pinned here.
critic_model = ChatOllama(
    model=CRITIC_MODEL_NAME,
    base_url=CRITIC_BASE_URL,
)
