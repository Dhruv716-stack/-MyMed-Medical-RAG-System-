from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import from config so chunk size is always in sync
# was: hardcoded 700 and 120 — ignored config.py entirely
from rag_pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP


def medical_rag_chunking(docs):

    # Stage 1 — section level splits
    # Breaks document into broad sections first
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n"
        ]
    )

    section_docs = section_splitter.split_documents(docs)

    # Stage 2 — embedding friendly final chunks
    # CHANGED: was chunk_size=700, chunk_overlap=120 hardcoded
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # 1000 from config
        chunk_overlap=CHUNK_OVERLAP,  # 200 from config
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    final_chunks = chunk_splitter.split_documents(section_docs)

    return final_chunks