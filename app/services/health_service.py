from qdrant_client import QdrantClient
from memory.database import engine

import requests


def health_check():

    qdrant_ok = True
    ollama_ok = True
    db_ok = True

    try:

        client = QdrantClient(
            path="./qdrant_data"
        )

        client.get_collections()

    except Exception:

        qdrant_ok = False

    try:

        requests.get(
            "http://localhost:11434",
            timeout=3
        )

    except Exception:

        ollama_ok = False

    try:

        conn = engine.connect()

        conn.close()

    except Exception:

        db_ok = False

    return {

        "status": "healthy",

        "qdrant": qdrant_ok,

        "memory_db": db_ok,

        "ollama": ollama_ok
    }