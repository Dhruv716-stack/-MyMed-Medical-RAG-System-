import re
import time

from app.core.singleton import get_pipeline

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# CITATION EXTRACTION
# ---------------------------------------------------------------------------
# The generation prompt asks the LLM to cite sources inline in the answer text
# as:  [Source: <filename> | Page: <X>]  (see generation/generate.py).
# The pipeline itself is unchanged — it still returns the answer as a string.
# Here (API layer only) we parse those inline tags into a structured list so
# the frontend can render a clean "Citations" section.
#
# This function is defensive on purpose: it must NEVER raise, so a parsing
# hiccup can never break the /chat response. If anything goes wrong it simply
# returns an empty list and the answer is unaffected.
# ---------------------------------------------------------------------------

# Matches: [Source: some file.pdf | Page: 12]  (page is optional / may be N/A)
_CITATION_RE = re.compile(
    r"\[Source:\s*(?P<source>.*?)\s*(?:\|\s*Page:\s*(?P<page>[^\]]*?))?\s*\]",
    re.IGNORECASE,
)


def extract_citations(answer):
    """
    Pull inline [Source: file | Page: X] tags out of the answer text into a
    de-duplicated list of {"source", "page"} dicts. Returns [] for anything
    that isn't a normal non-empty string, and never raises.
    """
    try:
        if not answer or not isinstance(answer, str):
            return []

        seen = set()
        citations = []

        for m in _CITATION_RE.finditer(answer):
            source = (m.group("source") or "").strip()
            page = (m.group("page") or "").strip()

            if not source:
                continue

            key = (source.lower(), page.lower())
            if key in seen:
                continue
            seen.add(key)

            citations.append({"source": source, "page": page or "N/A"})

        return citations

    except Exception:
        # Citations are a nice-to-have; never let them break the answer.
        return []


def chat(
    query: str,
    user_id: str,
    session_id: str
):
    """
    Main chat service.

    - Routes the query through the existing MedicalRAGPipeline, which
      internally decides general-chat vs medical-RAG and saves memory.
    - Passes user_id / session_id so each user's memory and uploads
      stay isolated (multi-tenant storage).
    - Measures latency.
    - Prevents API crashes from bubbling up to the client.
    """

    pipeline = get_pipeline()

    # Auto-name the chat from its first message (only renames a still-default
    # "New Chat" title; safe/best-effort, never raises). Done here so the
    # sidebar shows a meaningful title instead of "New Chat".
    try:
        from app.services.session_service import rename_session_if_default
        rename_session_if_default(
            session_id=session_id,
            user_id=user_id,
            first_message=query,
        )
    except Exception:
        pass

    start = time.time()

    try:

        answer = pipeline.run(
            query=query,
            user_id=user_id,
            session_id=session_id,
        )

        latency = round(
            time.time() - start,
            2
        )

        # Parse inline [Source: ... | Page: ...] tags into a structured list.
        # extract_citations() only reads strings and never raises, so a dict
        # answer (general-chat / error shapes) safely yields [].
        return {
            "answer": answer,
            "citations": extract_citations(answer),
            "latency": latency,
        }

    except Exception as e:

      raise HTTPException(
        status_code=500,
        detail=str(e)
      )
