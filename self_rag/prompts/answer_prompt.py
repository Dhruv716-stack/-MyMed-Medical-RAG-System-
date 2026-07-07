"""
self_rag.prompts.answer_prompt
==============================

Prompt definitions for the Answer Critic.

The Answer Critic evaluates whether a generated answer is
faithful to the retrieved medical evidence.

Responsibilities
----------------
- Verify grounding
- Detect hallucinations
- Estimate answer confidence
- Decide whether regeneration is required

The critic NEVER generates a new answer.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """
You are an expert Medical Answer Evaluation System.

Your ONLY responsibility is to evaluate the quality of an AI-generated
medical answer.

DO NOT rewrite the answer.

DO NOT provide medical advice.

Evaluate only whether the answer is supported by the retrieved evidence.

Return ONLY valid JSON.

Never output markdown.

Never explain outside JSON.
"""

JSON_SCHEMA = {
    "grounded": "boolean",
    "faithful": "boolean",
    "hallucination_detected": "boolean",
    "confidence": "float (0.0-1.0)",
    "retry_required": "boolean",
    "reasoning": "string",
    "missing_information": [
        "string"
    ]
}

USER_PROMPT_TEMPLATE = """
Evaluate the following answer.

=========================
User Query
=========================
{query}

=========================
Retrieved Medical Context
=========================
{context}

=========================
Generated Answer
=========================
{answer}

=========================
Evaluation Rules
=========================

Determine:

1. Is every medical claim supported?

2. Is anything hallucinated?

3. Is important information missing?

4. Should the answer be regenerated?

Return ONLY JSON.

Schema:

{schema}
"""


def build_answer_prompt(
    query: str,
    retrieved_chunks: list[str],
    generated_answer: str,
) -> list[dict[str, str]]:
    """
    Build Answer Critic prompt.

    Parameters
    ----------
    query
        Original user query.

    retrieved_chunks
        Retrieved document chunks.

    generated_answer
        LLM generated answer.

    Returns
    -------
    list[dict]
        Chat messages compatible with
        OpenAI / Groq / Ollama / LangChain.
    """

    context = "\n\n".join(
        f"[Chunk {i+1}]\n{text}"
        for i, text in enumerate(retrieved_chunks)
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        context=context,
        answer=generated_answer,
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