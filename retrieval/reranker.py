# retrieval/reranker.py

from typing import List, Sequence
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# -------------------------------
# Load model once (important)
# -------------------------------
_model = CrossEncoder("BAAI/bge-reranker-base")

# -------------------------------
# Rerank function
# -------------------------------
def rerank_documents(
    query: str,
    docs: Sequence,
    top_k: int = 5
) -> List[Document]:
    """
    Reranks documents based on query relevance.

    Accepts:
    - List[Document]
    - List[List[Document]] (e.g., [mmr_results, hybrid_reults])
    """

    if not docs:
        return []

    # Normalize input into a flat list of Document.
    flat_docs: List[Document] = []
    for item in docs:
        if isinstance(item, list):
            flat_docs.extend(item)
        else:
            flat_docs.append(item)

    flat_docs = [d for d in flat_docs if hasattr(d, "page_content")]
    if not flat_docs:
        return []

    # Step 1: Create pairs
    pairs = [(query, doc.page_content) for doc in flat_docs]

    # Step 2: Predict scores (batching for speed)
    scores = _model.predict(pairs, batch_size=16)

    # Step 3: Attach scores to docs
    scored_docs = list(zip(flat_docs, scores))

    # Optional (debugging)
    for doc, score in scored_docs:
        doc.metadata["rerank_score"] = float(score)

    # Step 4: Sort
    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    # Step 5: Return top_k
    return [doc for doc, _ in ranked_docs[:top_k]]