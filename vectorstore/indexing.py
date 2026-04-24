
from typing import List
from langchain_core.documents import Document
from embeddings.model import embedding
from vectorstore.qdrant_client import get_qdrant_client
from vectorstore.schema import COLLECTION_NAME, get_vector_config

from langchain_qdrant import QdrantVectorStore

def create_collection():
    client = get_qdrant_client()

    # Reuse embedding object created in earlier embedding cell.
    dim = len(embedding.embed_query("test"))
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=get_vector_config(dim)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

def index_documents(docs: List[Document]):
    client = get_qdrant_client()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding
    )

    vectorstore.add_documents(docs)
    print(f"Indexed {len(docs)} documents")