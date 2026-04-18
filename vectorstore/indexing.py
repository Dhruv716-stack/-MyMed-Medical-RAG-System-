
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import Qdrant

from embeddings.model import embedding
from vectorstore.qdrant_client import get_qdrant_client
from vectorstore.schema import COLLECTION_NAME, get_vector_config


def create_collection():

    client = get_qdrant_client()


    dim = len(embedding.embed_query("test"))

    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME not in existing:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=get_vector_config(dim)
        )

        print(f"Created collection: {COLLECTION_NAME}")

    else:
        print("Collection already exists")


def index_documents(docs: List[Document]):

    embedding_model = embedding
    client = get_qdrant_client()

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    vectorstore.add_documents(docs)

    print(f"Indexed {len(docs)} documents")