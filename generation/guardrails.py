from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# =========================================================
# DOMAIN CLASSIFIER
# LLM uses its own knowledge to judge intent
# =========================================================

_domain_model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

DOMAIN_CHECK_PROMPT = """
You are a classifier for a medical and health document assistant.

Determine if the following query is something a person would ask
when looking for information in medical, health, or scientific
health documents.

This includes anything a person might look up in:
- Medical textbooks or clinical guidelines
- Public health or environmental health documents
- Health safety, sanitation, or water quality standards
- Pharmaceutical, biological, or chemical health research
- Any scientific or technical topic related to human health

Return ONLY one word: YES or NO

Query: {query}
"""


def is_medical_query(query: str) -> bool:
    """
    Returns True if query is medical/health related.
    Uses open-ended LLM classification — no hardcoded keyword list.
    Handles any medical term including those not explicitly listed.
    """
    try:
        prompt = ChatPromptTemplate.from_template(DOMAIN_CHECK_PROMPT)
        chain  = prompt | _domain_model | StrOutputParser()
        result = chain.invoke({"query": query}).strip().upper()
        return "YES" in result
    except Exception:
        # If classifier fails for any reason, allow the query through
        # Better to allow a non-medical query than block a valid one
        return True


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
    Validates query through 3 checks in order:

    1. Input length    — rejects queries over 500 chars
    2. Prompt injection — blocks known attack phrases
    3. Domain scope    — LLM checks if medical/health related
                         Uses open-ended prompt so ANY valid
                         medical term passes through correctly
    """

    # Check 1 — input length
    if len(query.strip()) > MAX_QUERY_LENGTH:
        return False

    query_lower = query.lower()

    # Check 2 — prompt injection
    for phrase in BLOCKED_PHRASES:
        if phrase in query_lower:
            return False

    # Check 3 — domain scope
    # LLM-based — handles all medical terms naturally
    # No keyword list = no false blocks on valid medical queries
    if not is_medical_query(query):
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