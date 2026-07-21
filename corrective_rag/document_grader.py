"""
corrective_rag.document_grader
==============================

Lightweight binary relevance grader for retrieved documents.

Uses the local Ollama model (qwen2.5:1.5b) to grade each document
as "yes" (relevant) or "no" (irrelevant) relative to the user query.

This replaces Self-RAG's multi-step RetrievalCritic + AdaptiveRetriever
with a single, fast grading pass.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

logger = logging.getLogger(__name__)

# ── Grader model ──────────────────────────────────────────────────────────────
# Same lightweight local model used by contextual compression.
# Grading is a simple binary task so 1.5b is more than enough.
_grader_model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

# ── Grading prompt ────────────────────────────────────────────────────────────
# Outputs ONLY "yes" or "no". Any other response is treated as "no".
# Prompt is intentionally simple and lenient so the 1.5b model doesn't
# over-reject documents that have partial or indirect relevance.
_GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a relevance grader. "
        "Given a QUESTION and a DOCUMENT, decide if the document is useful.\n\n"
        "Answer 'yes' if the document contains ANY information related to the question, "
        "even partially or indirectly.\n"
        "Answer 'no' ONLY if the document is completely unrelated to the question.\n\n"
        "When in doubt, always answer 'yes'.\n"
        "Reply with ONLY 'yes' or 'no'."
    )),
    ("human", "QUESTION: {query}\n\nDOCUMENT:\n{document}\n\nRelevant:"),
])

_grading_chain = _GRADING_PROMPT | _grader_model | StrOutputParser()

# ── Concurrency ───────────────────────────────────────────────────────────────
MAX_GRADING_WORKERS = max(1, int(os.getenv("GRADING_MAX_WORKERS", "4")))


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GradingResult:
    """Result of grading a batch of documents."""

    relevant_docs: List[Document] = field(default_factory=list)
    irrelevant_docs: List[Document] = field(default_factory=list)
    relevance_ratio: float = 0.0

    @property
    def all_relevant(self) -> bool:
        return len(self.irrelevant_docs) == 0 and len(self.relevant_docs) > 0

    @property
    def none_relevant(self) -> bool:
        return len(self.relevant_docs) == 0

    @property
    def some_relevant(self) -> bool:
        return len(self.relevant_docs) > 0 and len(self.irrelevant_docs) > 0


# ── Grader class ──────────────────────────────────────────────────────────────

class DocumentGrader:
    """
    Grades retrieved documents for relevance to the query.

    Usage
    -----
        grader = DocumentGrader()
        result = grader.grade(query="What is diabetes?", docs=retrieved_docs)

        if result.none_relevant:
            # trigger web search
        elif result.some_relevant:
            # use relevant_docs + web search for gaps
        else:
            # all relevant, proceed with generation
    """

    def _grade_single(self, query: str, doc: Document) -> bool:
        """Grade a single document. Returns True if relevant."""

        try:
            response = _grading_chain.invoke({
                "query": query,
                "document": doc.page_content[:1500],  # cap to avoid token overflow
            }).strip().lower()

            return response.startswith("yes")

        except Exception as exc:
            # On failure, assume relevant (conservative — don't lose docs)
            logger.warning(
                "Grading failed for one doc; assuming relevant: %s", exc
            )
            return True

    def grade(
        self,
        query: str,
        docs: List[Document],
    ) -> GradingResult:
        """
        Grade all documents for relevance.

        Parameters
        ----------
        query
            The user's question.
        docs
            Retrieved documents to grade.

        Returns
        -------
        GradingResult
        """

        if not docs:
            return GradingResult()

        # Grade concurrently for speed (local Ollama, no rate limits)
        if len(docs) <= 1:
            grades = [self._grade_single(query, doc) for doc in docs]
        else:
            workers = min(MAX_GRADING_WORKERS, len(docs))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                grades = list(
                    pool.map(lambda d: self._grade_single(query, d), docs)
                )

        relevant = []
        irrelevant = []

        for doc, is_relevant in zip(docs, grades):
            if is_relevant:
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        total = len(docs)
        ratio = len(relevant) / total if total > 0 else 0.0

        logger.info(
            "Document grading: %d/%d relevant (%.1f%%)",
            len(relevant),
            total,
            ratio * 100,
        )

        return GradingResult(
            relevant_docs=relevant,
            irrelevant_docs=irrelevant,
            relevance_ratio=ratio,
        )
