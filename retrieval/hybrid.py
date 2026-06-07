from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from vectorstore.store import get_vectorstore

# =========================================================
# CONFIG
# =========================================================

VECTOR_K   = 15    # widened from 4
BM25_K     = 15    # widened from 6
HYBRID_CAP = 25    # max unique docs to pass forward



# BM25 SINGLETON
# Built once, reused for all queries — no per-query rebuild
_bm25_retriever: Optional[BM25Retriever] = None
_bm25_docs_hash: Optional[int]           = None


def get_bm25_retriever(docs: List[Document]) -> BM25Retriever:
    """
    Return a cached BM25 retriever.
    Rebuilds only if the document list has changed.
    """
    global _bm25_retriever, _bm25_docs_hash

    # Use length + first/last content hash as a cheap change detector
    docs_hash = hash((len(docs), docs[0].page_content[:50] if docs else ""))

    if _bm25_retriever is None or _bm25_docs_hash != docs_hash:
        _bm25_retriever = BM25Retriever.from_documents(docs)
        _bm25_retriever.k = BM25_K
        _bm25_docs_hash   = docs_hash

    return _bm25_retriever


# =========================================================
# HYBRID RETRIEVE
# =========================================================

def hybrid_retrieve(query: str, docs: List[Document]) -> List[Document]:
    """
    Hybrid retrieval: dense vector search + BM25 keyword search.
    BM25 index is cached — never rebuilt per query.
    """
    vectorstore   = get_vectorstore()
    bm25_retriever = get_bm25_retriever(docs)   # cached singleton

    vector_docs = vectorstore.similarity_search(query, k=VECTOR_K)
    bm25_docs   = bm25_retriever.invoke(query)

    combined = vector_docs + bm25_docs

    # Deduplicate preserving order
    seen       = set()
    unique_docs = []
    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:HYBRID_CAP]   # was [:VECTOR_K] — that was the funnel bug