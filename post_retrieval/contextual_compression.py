import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List

# ── PREVIOUS IMPORTS (v1 — keep for easy rollback) ────────────────────────────
# from langchain_classic.retrievers import ContextualCompressionRetriever
# from langchain_classic.retrievers.document_compressors import LLMChainExtractor
# from langchain_core.retrievers import BaseRetriever
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

# Compression is a summarization task — Ollama qwen2.5:3b handles it well
# Keeps Groq TPM free for generation only
compression_model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

# ── v3 CHANGE: bypass short chunks entirely ────────────────────────────────────
# v1 had no bypass — every chunk went through LLMChainExtractor regardless of size
# Short chunks (≤400 chars) don't need compression; compressing them risks losing content
_BYPASS_CHARS = 400
# ──────────────────────────────────────────────────────────────────────────────

# ── v3 CHANGE: conservative compression prompt ────────────────────────────────
# v1 used LLMChainExtractor which gave qwen a default aggressive "extract only relevant"
# prompt — caused over-compression and loss of key medical sentences.
# New prompt explicitly tells qwen: keep all medical facts, only remove pure filler,
# and when in doubt keep the sentence. Returns text verbatim (no paraphrasing).
_COMPRESSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a medical text filter. Given a QUESTION and a PASSAGE, "
        "return only the sentences from the PASSAGE that are relevant to the QUESTION. "
        "RULES:\n"
        "1. Keep ALL sentences containing medical facts, numbers, dosages, diagnoses, "
        "   symptoms, treatments, or definitions — even if only loosely related.\n"
        "2. Remove ONLY sentences that are completely unrelated filler with zero "
        "   connection to the question.\n"
        "3. When in doubt, KEEP the sentence.\n"
        "4. Return the kept sentences verbatim — do NOT paraphrase or summarize.\n"
        "5. If the entire passage is relevant, return it unchanged.\n"
        "6. Never return an empty response — always return at least one sentence."
    )),
    ("human", "QUESTION: {query}\n\nPASSAGE:\n{passage}\n\nFiltered passage:"),
])

_compression_chain = _COMPRESSION_PROMPT | compression_model | StrOutputParser()
# ──────────────────────────────────────────────────────────────────────────────


def _compress_single(query: str, doc: Document) -> Document:
    """Compress one document. Bypasses compression for short chunks (v3)."""
    text = doc.page_content

    # v3: skip compression for short chunks — nothing meaningful to cut
    # v1: no bypass, all chunks compressed regardless of size
    if len(text) <= _BYPASS_CHARS:
        return doc

    compressed_text = _compression_chain.invoke({
        "query":   query,
        "passage": text,
    }).strip()

    # Safety fallback: if model returns empty string, keep original content
    if not compressed_text:
        return doc

    return Document(
        page_content=compressed_text,
        metadata=doc.metadata,
    )


def compress_documents(query: str, retriever_func, docs: List[Document]) -> List[Document]:
    """
    v3 — Conservative compression (less aggressive than v1 LLMChainExtractor).
    Preserves medical facts and bypasses short chunks to keep grounding context intact.

    To revert to v1 (LLMChainExtractor):
        1. Restore the old imports at the top of this file
        2. Replace this function body with the original SimpleRetriever + ContextualCompressionRetriever code
    """
    if not docs:
        return []

    compressed = []
    for doc in docs:
        compressed.append(_compress_single(query, doc))

    return compressed


# ── v1 ORIGINAL compress_documents (keep for rollback reference) ───────────────
# def compress_documents(query: str, retriever_func, docs: List[Document]) -> List[Document]:
#
#     class SimpleRetriever(BaseRetriever):
#         def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
#             return retriever_func(query, None)
#
#     base_retriever = SimpleRetriever()
#     compressor     = LLMChainExtractor.from_llm(compression_model)  # ← aggressive default
#
#     compression_retriever = ContextualCompressionRetriever(
#         base_retriever=base_retriever,
#         base_compressor=compressor,
#     )
#
#     compressed_docs = compression_retriever.invoke(query)
#     return compressed_docs
# ──────────────────────────────────────────────────────────────────────────────