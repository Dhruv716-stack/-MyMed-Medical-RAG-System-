# generation/guardrails.py

from typing import List
from langchain_core.documents import Document


BLOCKED_TOPICS = [
    "ignore previous instructions",
    "bypass",
    "system prompt",
    "hack",
    "exploit",
]


MIN_CONTEXT_LENGTH = 50


def validate_query(query: str) -> bool:
    """
    Detect malicious or irrelevant queries.
    """

    query_lower = query.lower()

    for blocked in BLOCKED_TOPICS:
        if blocked in query_lower:
            return False

    return True


def validate_context(docs: List[Document]) -> bool:
    """
    Ensure retrieved context is sufficient.
    """

    if not docs:
        return False

    total_length = sum(len(doc.page_content) for doc in docs)

    return total_length >= MIN_CONTEXT_LENGTH


def format_context(docs: List[Document]) -> str:
    """
    Convert documents into formatted context.
    """

    context_parts = []

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[Document {i} | Source: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def safe_fallback_response() -> str:
    """
    Safe fallback if retrieval fails.
    """

    return (
        "I could not find enough reliable information "
        "in the provided documents to answer the question safely."
    )