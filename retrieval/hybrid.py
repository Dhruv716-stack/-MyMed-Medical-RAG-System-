from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from vectorstore.store import get_vectorstore


# =========================================================
# CONFIG
# =========================================================

VECTOR_K = 15
BM25_K = 15
HYBRID_CAP = 25


# =========================================================
# BM25 CACHE
# =========================================================

_bm25_retriever: Optional[BM25Retriever] = None
_bm25_docs_hash: Optional[int] = None


def get_bm25_retriever(
    docs: List[Document]
):

    global _bm25_retriever
    global _bm25_docs_hash

    # ----------------------------------
    # NO DOCS AVAILABLE
    # ----------------------------------

    if not docs:

        return None

    # ----------------------------------
    # DOC HASH
    # ----------------------------------

    docs_hash = hash(

        (
            len(docs),

            docs[0].page_content[:50]
        )
    )

    # ----------------------------------
    # REBUILD ONLY IF DOCS CHANGED
    # ----------------------------------

    if (

        _bm25_retriever is None

        or

        _bm25_docs_hash != docs_hash

    ):

        _bm25_retriever = BM25Retriever.from_documents(
            docs
        )

        _bm25_retriever.k = BM25_K

        _bm25_docs_hash = docs_hash

    return _bm25_retriever


# =========================================================
# HYBRID RETRIEVE
# =========================================================

def hybrid_retrieve(
    query: str,
    docs: List[Document]
) -> List[Document]:

    vectorstore = get_vectorstore()

    vector_docs = vectorstore.similarity_search(

        query,

        k=VECTOR_K
    )

    bm25_retriever = get_bm25_retriever(
        docs
    )

    # ----------------------------------
    # VECTOR ONLY MODE
    # ----------------------------------

    if bm25_retriever is None:

        return vector_docs

    # ----------------------------------
    # BM25
    # ----------------------------------

    bm25_docs = bm25_retriever.invoke(
        query
    )

    # ----------------------------------
    # COMBINE
    # ----------------------------------

    combined = (

        vector_docs +

        bm25_docs
    )

    seen = set()

    unique_docs = []

    for doc in combined:

        if doc.page_content not in seen:

            unique_docs.append(
                doc
            )

            seen.add(
                doc.page_content
            )

    return unique_docs[
        :HYBRID_CAP
    ]