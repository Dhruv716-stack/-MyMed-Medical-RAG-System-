
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "medical_rag_documents"

def get_vector_config(dim: int):
    """Returns vector configuration for the collection."""
    return VectorParams(
        size=dim,
        distance=Distance.COSINE
    )