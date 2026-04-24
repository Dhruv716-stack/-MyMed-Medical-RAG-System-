from qdrant_client import QdrantClient

_qdrant_client = None

def get_qdrant_client():
    """Return a session-local in-memory Qdrant client for pipeline testing."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
        print("Using in-memory Qdrant (session only)")
    return _qdrant_client

from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "medical_rag_documents"

def get_vector_config(dim: int):
    """Returns vector configuration for the collection."""
    return VectorParams(
        size=dim,
        distance=Distance.COSINE
    )