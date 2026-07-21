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
You are an expert medical document assistant with access to medical PDFs and verified web sources.

Your goal is to provide thoroughly, well explained, well-structured, and clinically accurate answers.

STRICT RULES:
1. Use ONLY information present in the provided context.
2. Do NOT add explanations, history, or facts not in the context.
3. Every point MUST be directly supported by the context.
4. Never hallucinate citations — only cite what is in the context.
5. For PDF/document sources, cite as: [Source: filename | Page: X]:point of interest in the document.
6. For web sources, cite as: [Source: title | URL]
7. If the context does not contain enough information, say:
   "The provided documents do not contain enough information to answer this question."
8. Write detailed, informative responses — not short bullet points.
   Explain concepts thoroughly like a knowledgeable medical professional would.also ask some relevant questions at the end of the response to continue the converastion professionally.
"""


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(docs: List[Document]) -> str:
    """
    Build formatted context string from retrieved documents.
    Separates PDF/document sources from web sources for clarity.
    """

    pdf_parts = []
    web_parts = []
    pdf_idx = 1
    web_idx = 1

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "N/A")
        url    = doc.metadata.get("url", "")
        title  = doc.metadata.get("title", "")

        if source == "web_search":
            web_parts.append(
                f"""
WEB SOURCE {web_idx}
TITLE: {title}
URL: {url}

CONTENT: {doc.page_content}
"""
            )
            web_idx += 1
        else:
            pdf_parts.append(
                f"""
DOCUMENT {pdf_idx}
SOURCE: {source}
PAGE: {page}

CONTENT: {doc.page_content}
"""
            )
            pdf_idx += 1

    sections = []
    if pdf_parts:
        sections.append("=== MEDICAL DOCUMENT SOURCES ===\n" + "\n\n".join(pdf_parts))
    if web_parts:
        sections.append("=== WEB SOURCES ===\n" + "\n\n".join(web_parts))

    return "\n\n".join(sections)


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
    Produces detailed, comprehensive responses with clear source attribution.
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
    # DETECT SOURCE TYPES
    # -----------------------------------

    has_pdf_docs = any(d.metadata.get("source", "") != "web_search" for d in docs)
    has_web_docs = any(d.metadata.get("source", "") == "web_search" for d in docs)

    # -----------------------------------
    # BUILD CONTEXT
    # -----------------------------------

    context = build_context(docs)

    # -----------------------------------
    # USER PROMPT (adaptive based on source types)
    # -----------------------------------

    if has_pdf_docs and has_web_docs:
        # Mixed sources — show clear sections
        format_instructions = """
Answer using ONLY the context above. Structure your response clearly:

## Answer

Provide a comprehensive, detailed explanation answering the question. 
Write in flowing paragraphs — explain concepts thoroughly like a knowledgeable medical professional would.
Include relevant medical details, mechanisms, and clinical context from the documents.

### From Medical Documents
Summarize key findings from the PDF/document sources with detailed explanations:
- [Detailed point with explanation] [Source: filename | Page: X]
- [Detailed point with explanation] [Source: filename | Page: X]

### From Web Sources  
Summarize additional context from web sources:
- [Detailed point with explanation] [Source: title | URL]
- [Detailed point with explanation] [Source: title | URL]

## Conclusion
Provide a thorough concluding summary integrating all sources.

**Sources Referenced:**
- List all document sources (filename, page) and web URLs used.

Note: This answer is based on the provided medical documents and verified web sources.
"""
    elif has_pdf_docs:
        # PDF only — standard medical document format
        format_instructions = """
Answer using ONLY the context above. Structure your response clearly:

## Answer

Provide a comprehensive, detailed explanation answering the question.
Write in flowing paragraphs — explain concepts thoroughly like a knowledgeable medical professional would.
Include relevant medical details, mechanisms, and clinical context from the documents.

### Key Findings
Provide detailed explanations for each key finding:
1. [Detailed explanation with medical context] [Source: filename | Page: X]
2. [Detailed explanation with medical context] [Source: filename | Page: X]
3. [Detailed explanation with medical context] [Source: filename | Page: X]

## Conclusion
Provide a thorough concluding summary based on the medical documents.

**Sources Referenced:**
- List all document sources with filename and page numbers.

Note: This answer is based solely on the provided medical documents.
"""
    else:
        # Web only
        format_instructions = """
Answer using ONLY the context above. Structure your response clearly:

## Answer

Provide a comprehensive, detailed explanation answering the question.
Write in flowing paragraphs — explain concepts thoroughly with medical accuracy.

### Key Points
Provide detailed explanations for each key point:
1. [Detailed explanation] [Source: title | URL]
2. [Detailed explanation] [Source: title | URL]
3. [Detailed explanation] [Source: title | URL]

## Conclusion
Provide a thorough concluding summary.

**Sources Referenced:**
- List all web URLs used in this answer.

Note: This answer is based on verified web sources.
"""

    user_prompt = f"""
CONTEXT:
{context}

QUESTION: {query}

{format_instructions}
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