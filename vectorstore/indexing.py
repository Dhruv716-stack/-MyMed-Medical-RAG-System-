from typing import List, Optional
from langchain_core.documents import Document

from embeddings.model import embedding

from vectorstore.qdrant_client import get_qdrant_client

from vectorstore.schema import (
    COLLECTION_NAME,
    get_vector_config
)

from vectorstore.cache import (
    get_file_hash,
    load_cache,
    save_cache
)

from rag_pipeline.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from langchain_qdrant import QdrantVectorStore


# =========================================================
# COLLECTION MANAGEMENT
# =========================================================

def create_collection(
    recreate: bool = False
):

    client = get_qdrant_client()

    dim = len(
        embedding.embed_query("test")
    )

    collections = {

        c.name

        for c in client.get_collections().collections
    }

    # ----------------------------------------
    # OPTIONAL DELETE
    # ----------------------------------------

    if recreate and COLLECTION_NAME in collections:

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            f"Deleted collection: {COLLECTION_NAME}"
        )

        collections.remove(
            COLLECTION_NAME
        )

    # ----------------------------------------
    # CREATE IF MISSING
    # ----------------------------------------

    if COLLECTION_NAME not in collections:

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=get_vector_config(dim)
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    else:

        print(
            f"Collection already exists: {COLLECTION_NAME}"
        )


# =========================================================
# VECTORSTORE
# =========================================================

def get_vectorstore():

    client = get_qdrant_client()

    return QdrantVectorStore(

        client=client,

        collection_name=COLLECTION_NAME,

        embedding=embedding
    )


# =========================================================
# SMART REINDEX CHECK
# =========================================================

def needs_reindex(
    file_path: str
) -> bool:

    pdf_hash = get_file_hash(
        file_path
    )

    cache = load_cache()

    # ----------------------------------------
    # COLLECTION DELETED
    # ----------------------------------------

    client = get_qdrant_client()

    collections = {

        c.name

        for c in client.get_collections().collections
    }

    if COLLECTION_NAME not in collections:

        return True

    # ----------------------------------------
    # FILE REGISTRY
    # ----------------------------------------

    indexed_files = cache.get(
        "indexed_files",
        {}
    )

    # ----------------------------------------
    # NEW PDF
    # ----------------------------------------

    if file_path not in indexed_files:

        indexed_files[file_path] = pdf_hash

        cache["indexed_files"] = indexed_files

        save_cache(cache)

        return True

    # ----------------------------------------
    # PDF MODIFIED
    # ----------------------------------------

    if indexed_files[file_path] != pdf_hash:

        indexed_files[file_path] = pdf_hash

        cache["indexed_files"] = indexed_files

        save_cache(cache)

        return True

    # ----------------------------------------
    # CONFIG CHANGED
    # ----------------------------------------

    current_config = {

        "embedding_model":
        EMBEDDING_MODEL,

        "chunk_size":
        CHUNK_SIZE,

        "chunk_overlap":
        CHUNK_OVERLAP
    }

    saved_config = cache.get(
        "config",
        {}
    )

    if current_config != saved_config:

        cache["config"] = current_config

        save_cache(cache)

        return True

    return False


# =========================================================
# INDEX DOCUMENTS
# =========================================================

def index_documents(
    docs: List[Document],
    file_path: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source_type: str = "default"
):

    if not docs:

        raise ValueError(
            "No documents provided for indexing."
        )

    # ----------------------------------------
    # ADD SOURCE METADATA
    # source_type defaults to "default" so old calls
    # (admin knowledge base ingestion) are unaffected.
    # user_id/session_id are only attached when given,
    # so user-uploaded chunks can later be filtered out
    # for everyone else at retrieval time.
    # ----------------------------------------

    for doc in docs:

        doc.metadata["source"] = file_path
        doc.metadata["source_type"] = source_type

        if user_id is not None:
            doc.metadata["user_id"] = user_id

        if session_id is not None:
            doc.metadata["session_id"] = session_id

    # ----------------------------------------
    # INDEX
    # ----------------------------------------

    vectorstore = get_vectorstore()

    vectorstore.add_documents(
        docs
    )

    print(
        f"Indexed {len(docs)} chunks "
        f"from {file_path}"
    )