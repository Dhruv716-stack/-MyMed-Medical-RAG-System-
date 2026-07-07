# pipeline.py

from typing import Dict
from pathlib import Path
import traceback
import time

from langsmith import trace, traceable
from sqlalchemy import false


# =========================================================
# INGESTION
# =========================================================

from ingestion.parser import fast_doc_loader
from ingestion.cleaner import clean_documents
from ingestion.chunker import medical_rag_chunking


# =========================================================
# VECTOR STORE
# =========================================================

from vectorstore.indexing import (
    index_documents,
    create_collection,
    needs_reindex
)


# =========================================================
# PRE RETRIEVAL
# =========================================================

from pre_retrieval.query_rewritter import rewrite_query

from pre_retrieval.multi_query import (
    generate_multi_queries
)

from pre_retrieval.ambiguity_detector import (
    is_ambiguous_llm
)


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

from post_retrieval.contextual_compression import (
    compress_documents
)


# =========================================================
# GENERATION
# =========================================================

from generation.generate import generate_answer

from generation.structured_output import clean_output
# =========================================================
# MEMORY
# =========================================================

from memory.memory_manager import (
    save_message,
    DEFAULT_USER,
    DEFAULT_SESSION
)

from memory.memory_builder import (
    build_memory_context
)

from memory.upload_manager import (
    save_uploaded_file,
    get_uploaded_files
)

from router.classifier import (
    classify_query
)

from chat.general_chat import (
    general_chat
)

# =========================================================
# MAIN PRODUCTION RAG PIPELINE
# =========================================================

class MedicalRAGPipeline:

    def __init__(
        self,
        verbose: bool = True
    ):

        self.verbose = verbose
        self.active_file = None

        # -----------------------------------
        # STORAGE
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
        self.chat_history = ""

        self.intent = ""
    # =====================================================
    # LOGGER
    # =====================================================

    def log(self, message: str):

        if self.verbose:
            print(message)


    # =====================================================
    # INGESTION
    # =====================================================

    @traceable(name="Document Ingestion")

    def ingest(
        self,
        file_path: str,
        force_reindex: bool = False,
        user_id: str = None,
        session_id: str = None,
        source_type: str = "default"
    ):

        with trace("Ingestion") as rt:

            start = time.time()

            self.log("\n" + "=" * 70)
            self.log("STEP 1 — DOCUMENT INGESTION")
            self.log("=" * 70)

            # -----------------------------------
            # LOAD
            # -----------------------------------

            self.raw_docs = list(
                fast_doc_loader(file_path)
            )

            self.log(
                f"\nLoaded Documents: {len(self.raw_docs)}"
            )

            # -----------------------------------
            # CLEAN
            # -----------------------------------

            self.cleaned_docs = clean_documents(
                self.raw_docs
            )

            self.log("Cleaning completed.")

            # -----------------------------------
            # CHUNK
            # -----------------------------------

            self.chunks = medical_rag_chunking(
                self.cleaned_docs
            )

            self.log(
                f"Generated Chunks: {len(self.chunks)}"
            )
            
            # -----------------------------------
            # SMART INDEXING
            # -----------------------------------
            
            
            # User uploads must ALWAYS be indexed, never skipped by the
            # file-hash cache. The cache keys on file path only, so the same
            # PDF uploaded by different users/sessions would otherwise be
            # skipped after the first time — leaving the new user's chunks
            # WITHOUT their user_id/session_id tags. Retrieval then filters
            # to "this user's chunks only" and finds nothing, producing
            # "documents do not contain enough information". Forcing indexing
            # for user_upload guarantees each upload gets its own tenant tags.
            is_user_upload = (source_type == "user_upload")

            should_reindex = (
               force_reindex
               or is_user_upload
               or needs_reindex(file_path)
            )

            if should_reindex:

               self.log(
              "\nIndex rebuild required..."
               )

               create_collection(
                 recreate=False
               )

               index_documents(
                  docs=self.chunks,
                  file_path=file_path,
                  user_id=user_id,
                  session_id=session_id,
                  source_type=source_type
                )

               self.log(
               "Vector indexing completed."
               )
               
            else:

              self.log(
               "Using existing index."
              )


            rt.add_metadata({

                "documents":
                len(self.raw_docs),

                "chunks":
                len(self.chunks),

                "smart_indexing":
                True
            })

            rt.add_outputs({

                "latency":
                round(time.time() - start, 2)
            })

        return self


    # =====================================================
    # QUERY PREPARATION
    # =====================================================

    @traceable(name="Query Preparation")

    def prepare_query(
        self,
        query: str
    ):

        with trace("Query Preparation") as rt:

            self.log("\n" + "=" * 70)
            self.log("STEP 2 — QUERY PREPARATION")
            self.log("=" * 70)

            try:

               rewritten_query = rewrite_query(query)  
            except Exception as e:
                self.log(
                    f"Rewrite failed: {e}"
                )

                rewritten_query = query

            queries = [rewritten_query]

            # -----------------------------------
            # MULTI QUERY
            # -----------------------------------

            if is_ambiguous_llm(rewritten_query):

                multi_queries = generate_multi_queries(
                    rewritten_query
                )

                cleaned_queries = []

                for q in multi_queries:

                    q = q.strip()

                    if q and q not in cleaned_queries:

                        cleaned_queries.append(q)

                queries.extend(cleaned_queries)

            queries = list(dict.fromkeys(queries))

            self.queries = queries

            self.log("\nGenerated Queries:\n")

            for i, q in enumerate(queries, 1):

                self.log(f"{i}. {q}")
            
            rt.add_metadata({

                "query":
                query,

                "rewritten_query":
                rewritten_query,

                "queries":
                queries
            })

        return self
    
       
    # =====================================================
    # RETRIEVAL
    # =====================================================

    @traceable(name="Retrieval")

    def retrieve(self, user_id: str = None, session_id: str = None, restrict_to_user_upload: bool = False):
        print(
            "Chunks available:",
            len(self.chunks)
        )

        with trace("Hybrid + MMR + Reranking") as rt:

            self.log("\n" + "=" * 70)
            self.log("STEP 3 — RETRIEVAL")
            self.log("=" * 70)

            all_results = []
           
            # -----------------------------------
            # HYBRID
            # Already loops all queries — unchanged
            # -----------------------------------
            for query in self.queries:

                results = hybrid_retrieve(
                    query=query,
                    docs=self.chunks,
                    user_id=user_id,
                    session_id=session_id,
                    restrict_to_user_upload=restrict_to_user_upload
                )

                all_results.extend(results)

            # -----------------------------------
            # REMOVE DUPLICATES
            # -----------------------------------

            seen        = set()
            unique_docs = []

            for doc in all_results:

                content = doc.page_content.strip()

                if content not in seen:
                    unique_docs.append(doc)
                    seen.add(content)

            self.hybrid_results = unique_docs

            self.log(
                f"Hybrid Retrieved Docs: {len(self.hybrid_results)}"
            )

            # -----------------------------------
            # MMR
            # CHANGED: now loops all queries
            # was: mmr_retriever.invoke(self.queries[0])
            # now: runs for every query, merges + deduplicates
            # -----------------------------------

            mmr_retriever = get_mmr_retriever(user_id=user_id, session_id=session_id, restrict_to_user_upload=restrict_to_user_upload)
            mmr_all       = []

            for query in self.queries:

                mmr_results = mmr_retriever.invoke(query)
                mmr_all.extend(mmr_results)

            # Deduplicate MMR results
            seen       = set()
            mmr_unique = []

            for doc in mmr_all:

                content = doc.page_content.strip()

                if content not in seen:
                    mmr_unique.append(doc)
                    seen.add(content)

            self.mmr_results = mmr_unique

            self.log(
                f"MMR Results: {len(self.mmr_results)}"
            )

            # -----------------------------------
            # COMBINE
            # -----------------------------------

            combined = (
                self.hybrid_results +
                self.mmr_results
            )

            final_docs = []
            seen       = set()

            for doc in combined:

                content = doc.page_content.strip()

                if content not in seen:
                    final_docs.append(doc)
                    seen.add(content)

            self.log(
                f"Combined Unique Docs: {len(final_docs)}"
            )

            # -----------------------------------
            # RERANK
            # CHANGED: top_k 5 → 10
            # Reranker scores against queries[0] — most precise
            # query version — this is correct and intentional
            # -----------------------------------

            self.reranked_docs = rerank_documents(

                query=self.queries[0],

                docs=final_docs,

                top_k=10                
            )

            self.log(
                f"Final Retrieved Docs: {len(self.reranked_docs)}"
            )

            rt.add_metadata({

                "hybrid_docs"  : len(self.hybrid_results),
                "mmr_docs"     : len(self.mmr_results),
                "combined_docs": len(final_docs),
                "reranked_docs": len(self.reranked_docs)
            })

            rt.add_outputs({

                "retrieved_docs": [
                    d.page_content[:200]
                    for d in self.reranked_docs
                ]
            })

        return self


    # =====================================================
    # POST RETRIEVAL
    # =====================================================

    @traceable(name="Post Retrieval")

    def post_retrieval(self):

        with trace("Filtering + Compression") as rt:

            self.log("\n" + "=" * 70)
            self.log("STEP 4 — POST RETRIEVAL")
            self.log("=" * 70)

            # -----------------------------------
            # FILTER
            # -----------------------------------

            self.filtered_docs = filter_documents(
                self.reranked_docs
            )

            # -----------------------------------
            # COMPRESS
            # -----------------------------------

            self.compressed_docs = compress_documents(

                query=self.queries[0],

                retriever_func=lambda q, _: (
                    self.filtered_docs
                ),

                docs=self.filtered_docs
            )

            if len(self.compressed_docs) < 3:
                self.log("Compression returned < 3 docs, falling back to filtered_docs[:5]")
                self.compressed_docs = self.filtered_docs[:5]

            self.log(
                f"Compressed Docs: {len(self.compressed_docs)}"
            )

            rt.add_metadata({

                "filtered_docs":
                len(self.filtered_docs),

                "compressed_docs":
                len(self.compressed_docs)
            })

        return self


    # =====================================================
    # GENERATION
    # =====================================================

    @traceable(name="Generation")

    def generate(self):

        with trace("Grounded Generation") as rt:

            self.log("\n" + "=" * 70)
            self.log("STEP 5 — GENERATION")
            self.log("=" * 70)

            raw_answer = generate_answer(

                query=self.queries[0],

                docs=self.compressed_docs
            )

            self.final_answer = clean_output(
                raw_answer
            )

            rt.add_outputs({

                "answer":
                self.final_answer
            })

        return self


    # =====================================================
    # FULL PIPELINE
    # =====================================================
    
    @traceable(name="Medical RAG Pipeline")
    
    def run(
    self,
    query,
    file_path:str=None,
    force_reindex:bool=False,
    debug:bool=False,
    user_id:str=None,
    session_id:str=None,
    source_type:str="default"
    ):
        try:

            with trace("Full Pipeline") as rt:
                start = time.time()

                memory_context = build_memory_context(
                    user_id=user_id,
                    session_id=session_id
                )
                
                intent = classify_query(
                   query
                )
                intent = intent.strip().upper()
                if "GENERAL_CHAT" in intent:
                    self.intent = "GENERAL_CHAT"

                elif "MEDICAL_RAG" in intent:
                    self.intent = "MEDICAL_RAG"

                else:
                    self.intent = "GENERAL_CHAT"

            # -----------------------------------
            # OPTIONAL INGESTION
            # -----------------------------------

                if file_path:
                    if self.active_file != file_path:
                        if force_reindex or needs_reindex(file_path):
                            self.ingest(
                            file_path=file_path,
                            force_reindex=force_reindex,
                            user_id=user_id,
                            session_id=session_id,
                            source_type=source_type
                    )
                    self.active_file = file_path

                    # Record this as a private upload so the user
                    # sees it again on session reopen, and so
                    # retrieval below is restricted to it only.
                    if source_type == "user_upload" and user_id and session_id:
                        save_uploaded_file(
                            source_path=file_path,
                            original_filename=Path(file_path).name,
                            user_id=user_id,
                            session_id=session_id
                        )

            # -----------------------------------
            # ROUTING: default KB vs user-upload KB
            # If this user already has uploaded file(s) in this
            # session, every answer comes only from their own
            # uploads — never the default knowledge base.
            # -----------------------------------

                restrict_to_user_upload = False

                if user_id and session_id:
                    if get_uploaded_files(user_id=user_id, session_id=session_id):
                        restrict_to_user_upload = True

            # -----------------------------------
            # QUERY PIPELINE
            # -----------------------------------

                if self.intent == "GENERAL_CHAT":

                  answer = general_chat(

                   query=query,

                   memory=memory_context
                  )

                  save_message(
                   role="user",

                   message=query,

                   user_id=user_id or DEFAULT_USER,

                   session_id=session_id or DEFAULT_SESSION
                  )

                  save_message(

                   role="assistant",

                   message=answer,

                   user_id=user_id or DEFAULT_USER,

                   session_id=session_id or DEFAULT_SESSION
                  )
                  return answer
            
            self.prepare_query(
               f"""
                Previous Conversation:

                {memory_context}

                Current Question:

                {query}
              """
            )

            self.retrieve(user_id=user_id, session_id=session_id, restrict_to_user_upload=restrict_to_user_upload)

            self.post_retrieval()

            self.generate()
            
            save_message(
                role="user",
                message=query,
                user_id=user_id or DEFAULT_USER,
                session_id=session_id or DEFAULT_SESSION
            )

            save_message(
                role="assistant",
                message=self.final_answer,
                user_id=user_id or DEFAULT_USER,
                session_id=session_id or DEFAULT_SESSION
            )

            rt.add_metadata({

                   "query":
                    query,

                   "ingested":
                    file_path is not None,
                    
                    "intent":
                    self.intent,

                   "history_length":
                    len(memory_context or ""),
                })

            rt.add_outputs({

                   "latency":
                   round(
                       time.time() - start,
                       2
                    ),

                    "answer":
                    self.final_answer
            })

            self.log("\n" + "=" * 70)
            self.log("PIPELINE COMPLETED")
            self.log("=" * 70)

        # -----------------------------------
        # UI MODE
        # -----------------------------------

            if not debug:

               return self.final_answer

        # -----------------------------------
        # DEBUG MODE
        # -----------------------------------

            return {

               "query":
                query,

                "generated_queries":
                self.queries,

                "retrieved_docs":
                self.reranked_docs,

                "compressed_docs":
                self.compressed_docs,

                "answer":
                self.final_answer
            }

        except Exception as e:

            traceback.print_exc()

            return {
               "error":
                str(e)
            }

    



    