from langchain_qdrant import QdrantVectorStore
from vectorstore.qdrant_client import get_qdrant_client
from vectorstore.schema import COLLECTION_NAME
from embeddings.model import embedding

def get_vectorstore():
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=embedding,
    )