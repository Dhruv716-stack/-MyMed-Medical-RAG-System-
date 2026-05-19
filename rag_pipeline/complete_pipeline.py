# pipeline.py

from typing import Dict

from langchain_core.documents import Document


# =========================================================
# INGESTION
# =========================================================

from ingestion.parser import fast_doc_loader
from ingestion.cleaner import clean_documents
from ingestion.chunker import medical_rag_chunking


# =========================================================
# VECTOR STORE
# =========================================================

from vectorstore.indexing import index_documents
from vectorstore.indexing import create_collection


# =========================================================
# PRE RETRIEVAL
# =========================================================

from pre_retrieval.query_rewritter import rewrite_query
from pre_retrieval.multi_query import generate_multi_queries
from pre_retrieval.ambiguity_detector import is_ambiguous_llm


# =========================================================
# RETRIEVAL
# =========================================================

from retrieval.hybrid import hybrid_retrieve
from retrieval.mmr import get_mmr_retriever
from retrieval.reranker import rerank_documents


# =========================================================
# POST RETRIEVAL
# =========================================================

from post_retrieval.filter import filter_documents
from post_retrieval.contextual_compression import compress_documents


# =========================================================
# GENERATION
# =========================================================

from generation.generate import generate_answer
from generation.structured_output import clean_output


# =========================================================
# MAIN PRODUCTION RAG PIPELINE
# =========================================================

class MedicalRAGPipeline:

    def __init__(self, verbose: bool = True):

        self.verbose = verbose

        # -----------------------------------
        # Storage Variables
        # -----------------------------------

        self.raw_docs = []
        self.cleaned_docs = []
        self.chunks = []

        self.queries = []

        self.hybrid_results = []
        self.mmr_results = []
        self.reranked_docs = []

        self.filtered_docs = []
        self.compressed_docs = []

        self.final_answer = ""

    # =====================================================
    # LOGGER
    # =====================================================

    def log(self, message: str):

        if self.verbose:
            print(message)

    # =====================================================
    # INGESTION
    # =====================================================

    def ingest(self, file_path: str):

        self.log("\n" + "=" * 70)
        self.log("STEP 1 — DOCUMENT INGESTION")
        self.log("=" * 70)

        # Parse document
        self.raw_docs = list(fast_doc_loader(file_path))

        self.log(f"\nLoaded Documents: {len(self.raw_docs)}")

        # Clean text
        self.cleaned_docs = clean_documents(
            self.raw_docs
        )

        self.log("Document cleaning completed.")

        # Chunking
        self.chunks = medical_rag_chunking(
            self.cleaned_docs
        )

        self.log(f"Generated Chunks: {len(self.chunks)}")
        
        create_collection()
        # Indexing
        index_documents(self.chunks)

        self.log("Vector store indexing completed.")

        return self

    # =====================================================
    # QUERY TRANSFORMATION
    # =====================================================

    def prepare_query(self, query: str):

        self.log("\n" + "=" * 70)
        self.log("STEP 2 — QUERY TRANSFORMATION")
        self.log("=" * 70)

        # Rewrite query
        rewritten_query = rewrite_query(query)

        self.log(f"\nRewritten Query:\n{rewritten_query}")

        queries = [rewritten_query]

        # Multi-query generation
        if is_ambiguous_llm(rewritten_query):

            self.log("\nAmbiguous query detected.")
            self.log("Generating multiple retrieval queries...")

            multi_queries = generate_multi_queries(
                rewritten_query
            )

            cleaned_queries = []

            for q in multi_queries:

                q = q.strip()

                if q and q not in cleaned_queries:
                    cleaned_queries.append(q)

            queries.extend(cleaned_queries)

        # Remove duplicates
        queries = list(dict.fromkeys(queries))

        self.queries = queries

        self.log("\nFinal Queries:\n")

        for i, q in enumerate(queries, 1):
            self.log(f"{i}. {q}")

        return self

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def retrieve(self):

        self.log("\n" + "=" * 70)
        self.log("STEP 3 — HYBRID RETRIEVAL")
        self.log("=" * 70)

        all_results = []

        # -----------------------------------
        # HYBRID RETRIEVAL
        # -----------------------------------

        self.log("\nRunning hybrid retrieval...")

        for query in self.queries:

            results = hybrid_retrieve(
                query=query,
                docs=self.chunks
            )

            all_results.extend(results)

        self.hybrid_results = all_results

        self.log(
            f"Retrieved Documents: {len(all_results)}"
        )

        # -----------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------

        unique_docs = []
        seen = set()

        for doc in all_results:

            content = doc.page_content.strip()

            if content not in seen:
                unique_docs.append(doc)
                seen.add(content)

        self.log(
            f"Unique Retrieved Docs: {len(unique_docs)}"
        )

        # -----------------------------------
        # MMR
        # -----------------------------------

        self.log("\nApplying MMR diversification...")

        mmr_retriever = get_mmr_retriever()

        self.mmr_results = mmr_retriever.invoke(
            self.queries[0]
        )

        self.log(
            f"MMR Results: {len(self.mmr_results)}"
        )

        # -----------------------------------
        # COMBINE RESULTS
        # -----------------------------------

        combined_docs = unique_docs + self.mmr_results

        deduplicated_docs = []
        seen = set()

        for doc in combined_docs:

            content = doc.page_content.strip()

            if content not in seen:
                deduplicated_docs.append(doc)
                seen.add(content)

        self.log(
            f"Combined Retrieval Docs: {len(deduplicated_docs)}"
        )

        # -----------------------------------
        # RERANKER
        # -----------------------------------

        self.log("\nApplying reranker...")

        self.reranked_docs = rerank_documents(
            query=self.queries[0],
            docs=deduplicated_docs,
            top_k=5
        )

        self.log(
            f"Top Reranked Docs: {len(self.reranked_docs)}"
        )

        return self

    # =====================================================
    # POST RETRIEVAL
    # =====================================================

    def post_retrieval(self):

        self.log("\n" + "=" * 70)
        self.log("STEP 4 — POST RETRIEVAL")
        self.log("=" * 70)

        # -----------------------------------
        # FILTERING
        # -----------------------------------

        self.filtered_docs = filter_documents(
            self.reranked_docs
        )

        self.log(
            f"\nFiltered Docs: {len(self.filtered_docs)}"
        )

        # -----------------------------------
        # CONTEXT COMPRESSION
        # -----------------------------------

        self.compressed_docs = compress_documents(
            query=self.queries[0],
            retriever_func=lambda q, _: self.filtered_docs,
            docs=self.filtered_docs
        )

        self.log(
            f"Compressed Docs: {len(self.compressed_docs)}"
        )

        return self

    # =====================================================
    # GENERATION
    # =====================================================

    def generate(self):

        self.log("\n" + "=" * 70)
        self.log("STEP 5 — GROUNDED GENERATION")
        self.log("=" * 70)

        # -----------------------------------
        # GENERATE RAW ANSWER
        # -----------------------------------

        raw_answer = generate_answer(
            query=self.queries[0],
            docs=self.compressed_docs
        )

        self.log("\nRaw answer generated.")

        # -----------------------------------
        # CLEAN + STRUCTURE OUTPUT
        # -----------------------------------

        self.final_answer = clean_output(
            raw_answer
        )

        self.log("Structured formatting applied.")

        return self

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def run(
        self,
        file_path: str,
        query: str
    ) -> Dict:

        (
            self
            .ingest(file_path)
            .prepare_query(query)
            .retrieve()
            .post_retrieval()
            .generate()
        )

        self.log("\n" + "=" * 70)
        self.log("MEDICAL RAG PIPELINE COMPLETED")
        self.log("=" * 70)

        return {

            "query": query,

            "generated_queries": self.queries,

            "retrieved_docs": self.reranked_docs,

            "compressed_docs": self.compressed_docs,

            "answer": self.final_answer
        }