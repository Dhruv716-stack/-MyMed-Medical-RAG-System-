import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
 )


# RAG PIPELINE IMPORTS


from rag_pipeline.complete_pipeline import MedicalRAGPipeline
from embeddings.model import embedding as hf_embeddings


# Ollama LLM (local)


try:
    from langchain_ollama import ChatOllama
except ImportError as exc:
    raise ImportError(
        "Missing langchain-ollama. Install with: pip install langchain-ollama"
    ) from exc

ollama_llm = ChatOllama(
    model="phi3:mini",
    temperature=0.0,
    base_url="http://localhost:11434",
 )


# Paths


workspace_root = Path.cwd()
project_root = workspace_root / "-MyMed-Medical-RAG-System-"
base_root = project_root if project_root.exists() else workspace_root

# Ensure project root is importable when running as a script.
if str(base_root) not in sys.path:
    sys.path.insert(0, str(base_root))

EVAL_DATASET_PATH = base_root / "evaluation" / "small_dataset.json"
EVAL_REPORT_PATH = base_root / "evaluation" / "reports" / "ragas_results.csv"


# Load Evaluation Dataset


def load_eval_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Run RAG Pipeline (project pipeline)


def run_rag_pipeline(eval_data, paths: list[str]):
    from ingestion.pipeline import ingestion_pipeline
    from vectorstore.indexing import create_collection, index_documents

    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    if not paths:
        raise ValueError("No input files provided for evaluation.")

    questions = []
    answers = []
    ground_truths = []
    contexts = []

    # Ingest and process chunks for all paths
    all_chunks = []
    for fp in paths:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        chunks = ingestion_pipeline(str(path))
        all_chunks.extend(chunks)

    # Index all chunks combined into Qdrant once
    create_collection()
    index_documents(all_chunks)

    rag = MedicalRAGPipeline(verbose=False)
    # Set internal state directly for sparse BM25 retrieval
    rag.chunks = all_chunks
    rag._index_built = True
    rag._indexed_file_path = "|".join(map(str, paths))

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        rag.prepare_query(question)
        rag.retrieve()
        rag.compressed_docs = rag.reranked_docs
        rag.generate()

        retrieved_docs = rag.reranked_docs
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        questions.append(question)
        answers.append(rag.final_answer)
        ground_truths.append(ground_truth)
        contexts.append(retrieved_contexts)

        print(f"Completed: {question}")

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


# Run RAGAS Evaluation


def run_ragas_evaluation(dataset_dict):
    dataset = Dataset.from_dict(dataset_dict)
    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result


# Save Results


def save_results(result, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    df.to_csv(path, index=False)
    print("\nEvaluation Results Saved")


# MAIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for the Medical RAG pipeline."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more document paths to index and evaluate.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Load dataset
    eval_data = load_eval_dataset(args.dataset)

    # 2. Run pipeline
    dataset_dict = run_rag_pipeline(eval_data, args.files)

    # 3. Evaluate
    result = run_ragas_evaluation(dataset_dict)

    # 4. Print scores
    print("\nFINAL SCORES\n")
    print(result)

    # 5. Save
    save_results(result, args.report)