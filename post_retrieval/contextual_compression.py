import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

load_dotenv()

# Compression is a summarization task — Ollama qwen2.5:3b handles it well
# Keeps Groq TPM free for generation only
compression_model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.0,
    base_url="http://localhost:11434",
)

def compress_documents(query: str, retriever_func, docs: List[Document]) -> List[Document]:

    class SimpleRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return retriever_func(query, None)

    base_retriever = SimpleRetriever()
    compressor     = LLMChainExtractor.from_llm(compression_model)  # ← Ollama now

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    compressed_docs = compression_retriever.invoke(query)
    return compressed_docs