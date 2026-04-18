from pathlib import Path
from langchain_core.documents import Document

import ingestion.parser as parser
from ingestion.cleaner import clean_documents
from ingestion.chunker import medical_rag_chunking


def ingestion_pipeline(file_path: str) -> list[Document]:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")


    docs = parser.fast_doc_loader(file_path)

    cleaned_docs = clean_documents(docs)

    chunks = medical_rag_chunking(cleaned_docs)

    processed_chunks = []

    for i, doc in enumerate(chunks):

        text = doc.page_content.strip()

        if not text:
            continue

        metadata = doc.metadata.copy()

        metadata["chunk_id"] = i
        metadata["doc_type"] = "medical_document"

        processed_chunks.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return processed_chunks
