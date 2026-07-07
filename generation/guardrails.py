from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


# =========================================================
# DOMAIN CLASSIFIER — REMOVED
# =========================================================
# A previous version used a local qwen2.5:3b LLM to classify whether a
# query was "medical" and blocked anything it judged non-medical. This
# backfired for a document-grounded RAG: valid questions whose answers
# ARE in the uploaded documents (communication in end-of-life care,
# ethics, advance directives, catchment-to-consumer risk management) were
# wrongly rejected and scored 0 faithfulness. Scope is now governed by
# retrieval + validate_context(): if the documents cover the question,
# context is retrieved and the question is answered; if not, the answer
# falls back gracefully. No LLM call is needed for query validation.


# =========================================================
# BLOCKED PHRASES — prompt injection detection
# =========================================================

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "bypass",
    "system prompt",
    "hack",
    "exploit",
    "jailbreak",
    "disregard instructions",
    "forget previous",
    "act as",
]


# =========================================================
# CONFIG
# =========================================================

MIN_CONTEXT_LENGTH = 50
MAX_QUERY_LENGTH   = 500


# =========================================================
# VALIDATE QUERY
# =========================================================

def validate_query(query: str) -> bool:
    """
    Validates query through 2 lightweight, LLM-free checks:

    1. Input length    — rejects queries over 500 chars
    2. Prompt injection — blocks known attack phrases

    NOTE: The previous "domain scope" check (an LLM classifier asking
    "is this medical?") was REMOVED. For a document-grounded RAG, scope
    is determined by what the retriever finds, not by the query's topic.
    A topic filter wrongly blocked valid questions whose answers ARE in
    the uploaded documents (e.g. communication, ethics, advance directives,
    catchment risk management). Relevance is now enforced by retrieval +
    validate_context() below, so off-topic queries simply retrieve no
    usable context and fall back gracefully — no LLM call required.
    """

    # Check 1 — input length
    if len(query.strip()) > MAX_QUERY_LENGTH:
        return False

    query_lower = query.lower()

    # Check 2 — prompt injection
    for phrase in BLOCKED_PHRASES:
        if phrase in query_lower:
            return False

    return True


# =========================================================
# VALIDATE CONTEXT
# =========================================================

def validate_context(docs: List[Document]) -> bool:
    """
    Ensures retrieved context is sufficient for generation.
    Falls back to safe response if context is too thin.
    """
    if not docs:
        return False

    total_length = sum(len(doc.page_content) for doc in docs)
    return total_length >= MIN_CONTEXT_LENGTH


# =========================================================
# FORMAT CONTEXT
# =========================================================

def format_context(docs: List[Document]) -> str:
    """
    Converts retrieved documents into a formatted context string.
    """
    context_parts = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# =========================================================
# RESPONSE HELPERS
# =========================================================

def safe_fallback_response() -> str:
    """Safe fallback if retrieved context is insufficient."""
    return (
        "I could not find enough reliable information "
        "in the provided documents to answer the question safely."
    )


def out_of_scope_response() -> str:
    """Response when query is not medical/health related."""
    return (
        "This system is designed to answer questions about medical "
        "and health documents only. "
        "Please ask a health or medicine related question."
    )


def query_too_long_response() -> str:
    """Response when query exceeds length limit."""
    return (
        f"Your query is too long. "
        f"Please keep your question under {MAX_QUERY_LENGTH} characters."
    )