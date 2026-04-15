from langchain_text_splitters import RecursiveCharacterTextSplitter


def medical_rag_chunking(docs):
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

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    final_chunks = chunk_splitter.split_documents(section_docs)

    return final_chunks