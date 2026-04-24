from qdrant_client import QdrantClient

_qdrant_client = None

def get_qdrant_client():
    """Return a session-local in-memory Qdrant client for pipeline testing."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
        print("Using in-memory Qdrant (session only)")
    return _qdrant_client

