from typing import List
from langchain_core.documents import Document

from retrieval.hybrid import hybrid_retrieve
from retrieval.mmr import get_mmr_retriever
from retrieval.reranker import rerank_documents

def retrieve_documents(
    query: str,
    docs: List[Document],
    top_k: int = 5
) -> List[Document]:
    """
    Main retrieval pipeline:
    1. Hybrid retrieval (vector + BM25)
    2. MMR retrieval (diversity)
    3. Combine + deduplicate
    4. Rerank
    """

    # -------------------------------
    # Step 1: Hybrid Retrieval
    # -------------------------------
    hybrid_docs = hybrid_retrieve(query, docs)

    # -------------------------------
    # Step 2: MMR Retrieval
    # -------------------------------
    mmr_retriever = get_mmr_retriever()
    mmr_docs = mmr_retriever.invoke(query)

    # -------------------------------
    # Step 3: Combine results
    # -------------------------------
    combined_docs = hybrid_docs + mmr_docs

    if not combined_docs:
        return []

    # -------------------------------
    # Step 4: Deduplicate
    # -------------------------------
    seen = set()
    unique_docs = []

    for doc in combined_docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    # -------------------------------
    # Step 5: Rerank
    # -------------------------------
    final_docs = rerank_documents(
        query=query,
        docs=unique_docs,
        top_k=top_k
    )

    return final_docs

