"""
self_rag.prompts.retrieval_prompt
=================================

Prompt definitions for the Retrieval Critic.

The Retrieval Critic evaluates the quality of retrieved
documents. It DOES NOT retrieve documents itself.

Responsibilities
----------------
- Evaluate retrieval quality
- Estimate retrieval confidence
- Determine context sufficiency
- Recommend adaptive retrieval
- Produce STRICT JSON output

No business logic should exist here.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """
You are an expert Medical Retrieval Evaluation System.

Your ONLY responsibility is to evaluate the quality of retrieved
medical evidence.

DO NOT answer the user's question.

DO NOT generate medical advice.

Instead evaluate:

1. Relevance of retrieved documents
2. Context completeness
3. Context sufficiency
4. Retrieval confidence
5. Whether another retrieval iteration is recommended

You MUST return ONLY valid JSON.

Never return markdown.

Never explain your reasoning outside JSON.
"""

JSON_SCHEMA = {
    "retrieval_confidence": "float (0.0-1.0)",
    "sufficient_context": "boolean",
    "suggested_top_k": "integer",
    "reasoning": "string",
    "reflection_metadata": {
        "relevance": "float",
        "coverage": "float",
        "reranker_quality": "float",
        "missing_topics": [
            "string"
        ]
    }
}

USER_PROMPT_TEMPLATE = """
Evaluate the following retrieval results.

=========================
User Query
=========================
{query}

=========================
Retrieved Context
=========================
{context}

=========================
Reranker Scores
=========================
{scores}

=========================
Evaluation Rules
=========================

Determine:

1. Are the retrieved documents relevant?

2. Are enough documents available?

3. Is important information missing?

4. Is another retrieval iteration recommended?

Return ONLY JSON matching this schema.

Schema:

{schema}
"""


def build_retrieval_prompt(
    query: str,
    retrieved_chunks: list[str],
    reranker_scores: list[float],
) -> list[dict[str, str]]:
    """
    Construct Retrieval Critic messages.

    Parameters
    ----------
    query
        User query.

    retrieved_chunks
        Retrieved chunk texts.

    reranker_scores
        Corresponding reranker scores.

    Returns
    -------
    list[dict]
        Chat messages compatible with
        OpenAI, Groq, Ollama and LangChain.
    """

    context = "\n\n".join(
        f"[Chunk {i + 1}]\n{text}"
        for i, text in enumerate(retrieved_chunks)
    )

    scores = "\n".join(
        f"Chunk {i + 1}: {score:.4f}"
        for i, score in enumerate(reranker_scores)
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
        scores=scores,
        schema=json.dumps(JSON_SCHEMA, indent=2),
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
        },
    ]
