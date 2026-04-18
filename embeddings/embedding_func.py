from embeddings.model import embedding
from langchain_core.documents import Document
from typing import List

def embed_docs(docs:List[Document])->List[List[float]]:
    
        texts=[doc.page_content for doc in docs]
        embeddings = embedding.embed_documents(texts)

        return embeddings


def embed_query(query:str)->List[float]:
    
    query = f"Represent this sentence for searching relevant passages: {query}"

    embedded_query = embedding.embed_query(query)

    return embedded_query