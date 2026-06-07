import re
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()


# =========================================================
# RAG PIPELINE IMPORTS
# =========================================================

from rag_pipeline.complete_pipeline import MedicalRAGPipeline
from embeddings.model import embedding as hf_embeddings


# =========================================================
# GROQ JUDGE LLM
# =========================================================

try:
    from langchain_groq import ChatGroq
except ImportError as exc:
    raise ImportError(
        "Missing langchain-groq. Install with: pip install langchain-groq"
    ) from exc

groq_judge = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_tokens=4096,
    api_key=os.getenv("GROQ_API_KEY"),
)


# =========================================================
# PATHS
# =========================================================

workspace_root = Path.cwd()
project_root   = workspace_root / "-MyMed-Medical-RAG-System-"
base_root      = project_root if project_root.exists() else workspace_root

if str(base_root) not in sys.path:
    sys.path.insert(0, str(base_root))

EVAL_DATASET_PATH = base_root / "evaluation" / "small_dataset.json"
EVAL_REPORT_PATH  = base_root / "evaluation" / "reports" / "ragas_results.csv"
CACHE_PATH        = base_root / "evaluation" / "dataset_dict_cache.json"


# =========================================================
# CONTEXT TRUNCATION
# =========================================================

MAX_CONTEXTS = 3
MAX_CHARS    = 600


def truncate_to_sentences(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate at sentence boundary — no mid-sentence cuts."""
    if len(text) <= max_chars:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    for sentence in sentences:
        if len(result) + len(sentence) <= max_chars:
            result += sentence + " "
        else:
            break

    if not result.strip():
        words = text[:max_chars].rsplit(' ', 1)
        result = words[0] if len(words) > 1 else text[:max_chars]

    return result.strip()


# =========================================================
# LOAD EVALUATION DATASET
# =========================================================

def load_eval_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# CACHE HELPERS
# =========================================================

def save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Pipeline cache saved → {path}")


def load_cache(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# RUN RAG PIPELINE
# =========================================================

def run_rag_pipeline(eval_data, paths: list) -> dict:
    from ingestion.pipeline import ingestion_pipeline
    from vectorstore.indexing import create_collection, index_documents

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    if not paths:
        raise ValueError("No input files provided for evaluation.")

    questions, answers, ground_truths, contexts = [], [], [], []

    # Ingest all PDFs once
    all_chunks = []
    for fp in paths:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        chunks = ingestion_pipeline(str(path))
        all_chunks.extend(chunks)

    # Index into Qdrant once
    create_collection()
    index_documents(all_chunks)

    rag = MedicalRAGPipeline(verbose=False)

    # Pass chunks to pipeline so hybrid_retrieve() can use them.
    # BM25 singleton in hybrid.py builds the index once on first
    # query and reuses it for all subsequent queries automatically.
    rag.chunks             = all_chunks
    rag._index_built       = True
    rag._indexed_file_path = "|".join(map(str, paths))

    for item in eval_data:
        question     = item["question"]
        ground_truth = item["ground_truth"]

        rag.prepare_query(question)
        rag.retrieve()
        rag.post_retrieval()
        rag.generate()

        retrieved_contexts = [
            doc.page_content for doc in rag.compressed_docs
        ]

        questions.append(question)
        answers.append(rag.final_answer)
        ground_truths.append(ground_truth)
        contexts.append(retrieved_contexts)

        print(f"Completed: {question}")

    dataset_dict = {
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    }

    save_cache(dataset_dict, CACHE_PATH)

    return dataset_dict


# =========================================================
# RUN RAGAS EVALUATION
# =========================================================

def run_ragas_evaluation(dataset_dict: dict):

    # Apply sentence-aware truncation before scoring
    truncated_contexts = [
        [
            truncate_to_sentences(ctx, max_chars=MAX_CHARS)
            for ctx in ctxs[:MAX_CONTEXTS]
        ]
        for ctxs in dataset_dict["contexts"]
    ]

    truncated_dict = {
        "question"    : dataset_dict["question"],
        "answer"      : dataset_dict["answer"],
        "contexts"    : truncated_contexts,
        "ground_truth": dataset_dict["ground_truth"],
    }

    dataset   = Dataset.from_dict(truncated_dict)
    ragas_llm = LangchainLLMWrapper(groq_judge)
    ragas_emb = LangchainEmbeddingsWrapper(hf_embeddings)

    # Assign LLM to each metric explicitly
    faithfulness.llm            = ragas_llm
    answer_relevancy.llm        = ragas_llm
    context_precision.llm       = ragas_llm
    context_recall.llm          = ragas_llm
    answer_relevancy.embeddings = ragas_emb

    # Sequential execution — prevents parallel Groq rate limit errors
    run_cfg = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=3,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    return result


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    df.to_csv(path, index=False)
    print(f"\nEvaluation Results Saved → {path}")


# =========================================================
# MAIN
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for the Medical RAG pipeline."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more PDF paths to index and evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVAL_DATASET_PATH,
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORT_PATH,
        help="Path to save the evaluation report CSV.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load pipeline results from cache instead of re-running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Load eval questions
    eval_data = load_eval_dataset(args.dataset)

    # 2. Run pipeline or load from cache
    if args.use_cache and CACHE_PATH.exists():
        print(f"Loading pipeline cache from {CACHE_PATH}")
        dataset_dict = load_cache(CACHE_PATH)
    else:
        dataset_dict = run_rag_pipeline(eval_data, args.files)

    # 3. Evaluate
    result = run_ragas_evaluation(dataset_dict)

    # 4. Print scores
    print("\nFINAL SCORES\n")
    print(result)

    # 5. Save
    save_results(result, args.report)