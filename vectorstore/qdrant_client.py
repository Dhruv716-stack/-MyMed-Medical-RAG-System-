from qdrant_client import QdrantClient

_qdrant_client = None

def get_qdrant_client():
  
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path="./qdrant_data")
        print("Using persistent Qdrant storage.")
    return _qdrant_client

