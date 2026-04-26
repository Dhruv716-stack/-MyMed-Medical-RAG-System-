
from typing import List, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from generation.retrieve_model import model


class CustomRetrieverWrapper(BaseRetriever):

    retriever_func: Callable
    docs: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever_func(query, self.docs)


def compress_documents(
    query: str,
    retriever_func: Callable,
    docs: List[Document]
) -> List[Document]:

    base_retriever = CustomRetrieverWrapper(
        retriever_func=retriever_func,
        docs=docs
    )

    compressor = LLMChainExtractor.from_llm(model)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    compressed_docs = compression_retriever.invoke(query)

    return compressed_docs