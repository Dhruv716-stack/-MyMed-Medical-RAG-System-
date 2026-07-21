"""
chat.general_chat
=================

General chat handler with optional web search enhancement.

When the query appears to be a factual or current-events question,
a web search is triggered to supplement the LLM's response with
up-to-date information.
"""

from generation.retrieve_model import model
from tools.web_search import web_search


# ── Heuristic: does the query look like it needs web info? ────────────────────
_FACTUAL_KEYWORDS = {
    "what is", "who is", "when did", "where is", "how does",
    "how many", "how much", "latest", "current", "recent",
    "today", "2024", "2025", "2026", "statistics", "population",
    "news", "update", "guidelines", "recommend",
}


def _needs_web_search(query: str) -> bool:
    """Simple heuristic: if the query contains factual/current keywords, search."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _FACTUAL_KEYWORDS)


def _format_web_results(results) -> str:
    """Format web search results into context text."""
    if not results:
        return ""

    parts = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        parts.append(
            f"[Web Result {i}] {title}\n"
            f"URL: {url}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


def general_chat(
    query,
    memory
):
    # Optionally enhance with web search
    web_context = ""
    if _needs_web_search(query):
        results = web_search(query, max_results=3)
        web_context = _format_web_results(results)

    web_section = ""
    if web_context:
        web_section = f"""
Here is some relevant information from the web that may help you give a more accurate and up-to-date response:

{web_context}

Use this web information to supplement your response when relevant.
"""

    prompt = f"""
give a helpful and elaborated response to the user query giving information about the query in detail.and also use the previous conversation to give a more accurate and helpful response to the user query.also dont report user about no previous data if you dont have any previous data just give a response to the user query without using any previous data if you dont have any previous data.
Previous conversation:

{memory}
{web_section}
User:

{query}
"""

    response = model.invoke(
        prompt
    )

    return response.content