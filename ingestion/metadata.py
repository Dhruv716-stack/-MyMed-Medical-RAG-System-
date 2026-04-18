# rag/ingestion/metadata.py
import uuid
from langchain_core.documents import Document


def add_metadata(chunks: list[Document], user_id: str = None) -> list[Document]:
    """
    Enrich chunks with structured metadata
    """

    enriched_docs = []
    total_chunks = len(chunks)

    for idx, doc in enumerate(chunks):

        # Copy existing metadata (VERY IMPORTANT)
        metadata = dict(doc.metadata) if doc.metadata else {}

        # Add core metadata
        metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": idx,
            "total_chunks": total_chunks,
        })

        # Optional but important for production
        if user_id:
            metadata["user_id"] = user_id

        enriched_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    return enriched_docs
