from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )

        print("Using Docker Qdrant server.")

    return _qdrant_client