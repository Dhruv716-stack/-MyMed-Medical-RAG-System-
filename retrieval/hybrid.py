from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from vectorstore.store import get_vectorstore

VECTOR_K = 4
BM25_K = 6

def build_bm25_retriever(docs: List[Document]):

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = BM25_K

    return retriever


def hybrid_retrieve(query: str, docs: List[Document]) -> List[Document]:

    vectorstore = get_vectorstore()
    bm25_retriever = build_bm25_retriever(docs)

    vector_docs = vectorstore.similarity_search(query, k=VECTOR_K)

    bm25_docs = bm25_retriever.invoke(query)


    combined_docs = vector_docs + bm25_docs


    seen = set()
    unique_docs = []

    for doc in combined_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:VECTOR_K]