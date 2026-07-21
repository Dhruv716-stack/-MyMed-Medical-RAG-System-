from typing import List, Optional, Dict, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from qdrant_client.models import Filter, FieldCondition, MatchValue

from vectorstore.store import get_vectorstore


# =========================================================
# CONFIG
# =========================================================

VECTOR_K = 15
BM25_K = 15
HYBRID_CAP = 25


# =========================================================
# BM25 CACHE
# ----------------------------------------------------------
# OLD (v1, single global -- caused cross-user leakage because
# every user/session shared one BM25 index built from whoever
# ingested last):
#
# _bm25_retriever: Optional[BM25Retriever] = None
# _bm25_docs_hash: Optional[int] = None
#
# def get_bm25_retriever(docs):
#     global _bm25_retriever, _bm25_docs_hash
#     if not docs:
#         return None
#     docs_hash = hash((len(docs), docs[0].page_content[:50]))
#     if _bm25_retriever is None or _bm25_docs_hash != docs_hash:
#         _bm25_retriever = BM25Retriever.from_documents(docs)
#         _bm25_retriever.k = BM25_K
#         _bm25_docs_hash = docs_hash
#     return _bm25_retriever
#
# NEW (v2): cache keyed by session_id, so each chat session gets
# its own BM25 index instead of sharing one global instance.
# ----------------------------------------------------------

_bm25_cache: Dict[str, Tuple[BM25Retriever, int]] = {}

_DEFAULT_BM25_KEY = "local_user::chat_1"


def get_bm25_retriever(
    docs: List[Document],
    session_key: str = _DEFAULT_BM25_KEY
):

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

    cached = _bm25_cache.get(session_key)

    # ----------------------------------
    # REBUILD ONLY IF DOCS CHANGED
    # ----------------------------------

    if (

        cached is None

        or

        cached[1] != docs_hash

    ):

        retriever = BM25Retriever.from_documents(
            docs
        )

        retriever.k = BM25_K

        _bm25_cache[session_key] = (retriever, docs_hash)

        return retriever

    return cached[0]


# =========================================================
# TENANT FILTER
# ----------------------------------------------------------
# Two modes:
#   restrict_to_user_upload=False (default) -> sees default KB
#     chunks OR this user's own uploaded chunks. Used when the
#     user has NOT uploaded anything for this session.
#   restrict_to_user_upload=True -> sees ONLY this user's own
#     uploaded chunks, default KB excluded entirely. Used once
#     the user has an active upload, per the product requirement
#     that uploaded-PDF answers must come only from that PDF.
# When user_id/session_id are not given (old callers), no filter
# is applied and behavior is unchanged from before this change.
# =========================================================

def _build_tenant_filter(
    user_id: Optional[str],
    session_id: Optional[str],
    restrict_to_user_upload: bool = False
) -> Optional[Filter]:

    if user_id is None or session_id is None:

        return None

    own_upload_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            ),
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=session_id)
            )
        ]
    )

    if restrict_to_user_upload:

        return own_upload_filter

    return Filter(

        should=[

            FieldCondition(
                key="metadata.source_type",
                match=MatchValue(value="default")
            ),

            own_upload_filter
        ]
    )


# =========================================================
# HYBRID RETRIEVE
# =========================================================

def hybrid_retrieve(
    query: str,
    docs: List[Document],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    restrict_to_user_upload: bool = False
) -> List[Document]:

    vectorstore = get_vectorstore()

    tenant_filter = _build_tenant_filter(
        user_id,
        session_id,
        restrict_to_user_upload=restrict_to_user_upload
    )

    vector_docs = vectorstore.similarity_search(

        query,

        k=VECTOR_K,

        filter=tenant_filter
    )

    bm25_key = (

        f"{user_id}::{session_id}"

        if user_id is not None and session_id is not None

        else _DEFAULT_BM25_KEY
    )

    bm25_retriever = get_bm25_retriever(
        docs,
        session_key=bm25_key
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