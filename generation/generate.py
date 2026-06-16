from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from generation.retrieve_model import model
from generation.guardrails import (
    validate_query,
    validate_context,
    safe_fallback_response,
    out_of_scope_response,
    query_too_long_response,
    BLOCKED_PHRASES,
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a medical document assistant.

STRICT RULES:
1. Use ONLY information present in the provided context.
2. Do NOT add explanations, history, or facts not in the context.
3. Every point MUST be directly supported by the context.
4. Never hallucinate citations — only cite what is in the context.
5. Citation format: [Source: filename | Page: X]
6. If the context does not contain enough information, say:
   "The provided documents do not contain enough information to answer this question."
"""


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(docs: List[Document]) -> str:
    """
    Build formatted context string from retrieved documents.
    Each document includes its source and page metadata.
    """

    context_parts = []

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"""
DOCUMENT {i}
SOURCE: {source}
PAGE: {page}

CONTENT: {doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    query: str,
    docs: List[Document]
) -> str:
    """
    Production-grade grounded answer generation.
    Answers are strictly grounded to retrieved context only.
    """

    # -----------------------------------
    # QUERY VALIDATION
    # -----------------------------------

    if len(query.strip()) > 500:
     return query_too_long_response()

    if not validate_query(query):
        query_lower = query.lower()
        if any(p in query_lower for p in BLOCKED_PHRASES):
            return "Query blocked by safety guardrails."
        return out_of_scope_response()


    # -----------------------------------
    # CONTEXT VALIDATION
    # -----------------------------------

    if not validate_context(docs):
        return safe_fallback_response()

    # -----------------------------------
    # BUILD CONTEXT
    # -----------------------------------

    context = build_context(docs)

    # -----------------------------------
    # USER PROMPT
    # -----------------------------------

    user_prompt = f"""
CONTEXT:
{context}

QUESTION: {query}

Answer using ONLY the context above. Use this format:

Introduction:
[1-2 sentences directly answering the question — only from context]

Key Points:
1. [point directly from context] [Source: filename | Page: X]
2. [point directly from context] [Source: filename | Page: X]
3. [point directly from context] [Source: filename | Page: X]

[Also include a conclusion based on the key points and generate a final answer.]

Note: This answer is based solely on the provided medical documents.
"""

    # -----------------------------------
    # GENERATE
    # -----------------------------------

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = model.invoke(messages)

    return response.content