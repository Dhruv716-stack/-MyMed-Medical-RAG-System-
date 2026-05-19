# generation/generate.py

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from generation.retrieve_model import model
from generation.guardrails import (
    validate_query,
    validate_context,
    safe_fallback_response,
)


SYSTEM_PROMPT = """
You are an expert medical document assistant.

STRICT RULES:
1. Answer ONLY from the provided context.
2. Never use outside knowledge.
3. If information is missing, say:
   "The provided documents do not contain enough information."
4. Generate detailed explanatory answers.
5. Use professional formatting.
6. Minimum 3 well-explained points.
7. Every important statement MUST contain citations.
8. Citation format:
   [Source: filename | Page: X]
9. Never hallucinate citations.
10. Use only citations actually present in context.
11. Add a short conclusion section.
"""


def build_context(docs: List[Document]) -> str:
    """
    Build formatted context with metadata citations.
    """

    context_parts = []

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"""
DOCUMENT {i}

SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


def generate_answer(
    query: str,
    docs: List[Document]
) -> str:
    """
    Production-grade grounded answer generation.
    """

    
    # Query Validation
    

    if not validate_query(query):
        return "Query blocked by safety guardrails."

   
    # Context Validation
    

    if not validate_context(docs):
        return safe_fallback_response()

    
    # Build Context
    

    context = build_context(docs)

    
    # User Prompt
    

    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

Generate a professional grounded answer.

Answer Requirements:

1. Add a short introduction.
2. Provide at least 3 detailed numbered points.
3. Every point must include citations.
4. Use this citation format:
   [Source: filename | Page: X]
5. Add a short conclusion.
6. Use ONLY provided context.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    
    # Generate
   

    response = model.invoke(messages)

    return response.content