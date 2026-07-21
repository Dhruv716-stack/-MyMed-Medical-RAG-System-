"""
tools.web_search
================

Lightweight web search utility using DuckDuckGo (no API key required).

Returns results as LangChain Document objects so they plug straight into
the existing retrieval / generation stack.

Usage
-----
    from tools.web_search import web_search

    docs = web_search("latest diabetes treatment guidelines", max_results=3)
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def web_search(
    query: str,
    *,
    max_results: int = 3,
    region: str = "wt-wt",
) -> List[Document]:
    """
    Search the web via DuckDuckGo and return results as Documents.

    Parameters
    ----------
    query
        Search query string.
    max_results
        Maximum number of results to return (default 3).
    region
        DuckDuckGo region code (default "wt-wt" = worldwide).

    Returns
    -------
    List[Document]
        Each document has:
        - page_content = search snippet
        - metadata = {"title": ..., "url": ..., "source": "web_search"}
    """

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error(
                "Neither 'ddgs' nor 'duckduckgo-search' is installed. "
                "Run: pip install ddgs"
            )
            return []

    documents: List[Document] = []

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    region=region,
                    max_results=max_results,
                )
            )

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            href = result.get("href", "")

            if not body:
                continue

            documents.append(
                Document(
                    page_content=body,
                    metadata={
                        "title": title,
                        "url": href,
                        "source": "web_search",
                    },
                )
            )

        logger.info(
            "Web search returned %d results for: %s",
            len(documents),
            query[:80],
        )

    except Exception as exc:
        logger.warning(
            "Web search failed (query=%s): %s",
            query[:80],
            exc,
        )

    return documents
