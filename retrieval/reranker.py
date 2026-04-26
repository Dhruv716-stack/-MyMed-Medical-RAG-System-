from typing import List, Sequence
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch

# Pick an available device to avoid CUDA assertion on CPU-only PyTorch.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CrossEncoder(
    "BAAI/bge-reranker-base",
    device=_device,
 )

MAX_CANDIDATES = 15
MAX_CHARS = 400
BATCH_SIZE = 32


def rerank_documents(
    query: str,
    docs: Sequence,
    top_k: int = 5
) -> List[Document]:

    if not docs:
        return []

    # -----------------------
    # Flatten input
    # -----------------------
    flat_docs: List[Document] = []
    for item in docs:
        if isinstance(item, list):
            flat_docs.extend(item)
        else:
            flat_docs.append(item)

    # -----------------------
    # Filter valid docs
    # -----------------------
    flat_docs = [d for d in flat_docs if hasattr(d, "page_content")]
    if not flat_docs:
        return []

    # -----------------------
    # LIMIT INPUT SIZE
    # -----------------------
    flat_docs = flat_docs[:MAX_CANDIDATES]

    # -----------------------
    # TRUNCATE TEXT
    # -----------------------
    pairs = [
        (query, doc.page_content[:MAX_CHARS])
        for doc in flat_docs
    ]

    # -----------------------
    # BATCH INFERENCE
    # -----------------------
    scores = _model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    # -----------------------
    # Sort
    # -----------------------
    ranked_docs = sorted(
        zip(flat_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked_docs[:top_k]]