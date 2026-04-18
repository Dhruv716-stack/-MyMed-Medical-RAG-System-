
import os
from qdrant_client import QdrantClient


def get_qdrant_client():
   
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")

    client = QdrantClient(
        url=url,
        api_key=api_key
    )

    return client