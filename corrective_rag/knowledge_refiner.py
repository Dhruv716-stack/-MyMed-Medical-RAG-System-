"""
corrective_rag.knowledge_refiner
================================

Extracts and distills relevant information from graded documents.

Takes the relevant documents (after grading) and refines them into
a focused knowledge context for the generator. This replaces the
separate contextual compression + answer reflection stages from Self-RAG.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Refiner model ─────────────────────────────────────────────────────────────
# Same local model as compression / grading — keeps Groq/Cerebras free.
_refiner_model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

# ── Refinement prompt ─────────────────────────────────────────────────────────
# Extracts key information from multiple documents into a coherent context.
_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a medical knowledge extractor. Given a QUESTION and multiple "
        "DOCUMENTS, extract and organize all relevant information into a clear, "
        "concise knowledge summary.\n\n"
        "RULES:\n"
        "1. Include ALL relevant medical facts, numbers, dosages, and key details.\n"
        "2. Organize information logically — group related facts together.\n"
        "3. Preserve source attribution (document numbers) for traceability.\n"
        "4. Remove duplicate information across documents.\n"
        "5. Do NOT add information not present in the documents.\n"
        "6. Do NOT generate an answer — only extract and organize facts.\n"
        "7. If documents contain conflicting information, include both with a note."
    )),
    ("human", "QUESTION: {query}\n\nDOCUMENTS:\n{documents}\n\nExtracted knowledge:"),
])

_refine_chain = _REFINE_PROMPT | _refiner_model | StrOutputParser()


class KnowledgeRefiner:
    """
    Refines relevant documents into focused knowledge context.

    Usage
    -----
        refiner = KnowledgeRefiner()
        refined_docs = refiner.refine(query="...", docs=relevant_docs)
    """

    def refine(
        self,
        query: str,
        docs: List[Document],
    ) -> List[Document]:
        """
        Refine documents by extracting and organizing relevant knowledge.

        For small document sets (≤3 docs), returns them as-is since
        grading already filtered irrelevant content.

        For larger sets, runs knowledge extraction to distill into
        a focused context.

        Parameters
        ----------
        query
            The user's question.
        docs
            Relevant documents (already graded).

        Returns
        -------
        List[Document]
            Refined documents ready for generation.
        """

        if not docs:
            return []

        # Small sets don't need refinement — grading was enough
        if len(docs) <= 3:
            logger.info("Skipping refinement for %d docs (≤3)", len(docs))
            return docs

        # Build document text for the refiner
        doc_texts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            doc_texts.append(
                f"[Document {i} | Source: {source} | Page: {page}]\n"
                f"{doc.page_content}"
            )

        documents_str = "\n\n---\n\n".join(doc_texts)

        try:
            refined_text = _refine_chain.invoke({
                "query": query,
                "documents": documents_str,
            }).strip()

            if not refined_text:
                logger.warning("Refinement returned empty; using original docs")
                return docs

            # Return as a single consolidated Document + keep originals
            # for source attribution
            refined_doc = Document(
                page_content=refined_text,
                metadata={
                    "source": "knowledge_refinement",
                    "original_doc_count": len(docs),
                },
            )

            # Return refined doc first, then top 2 originals for citation
            return [refined_doc] + docs[:2]

        except Exception as exc:
            logger.warning(
                "Knowledge refinement failed; using original docs: %s", exc
            )
            return docs
