
from qdrant_client.http.models import (
    Distance,
    VectorParams
)

COLLECTION_NAME = "medical_rag_documents"


def get_vector_config(embedding_dim: int):
    return VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE
    )