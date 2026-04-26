from typing import List
from langchain_core.documents import Document


MIN_CONTENT_LENGTH = 30
MAX_DOCS = 6


def filter_documents(docs: List[Document]) -> List[Document]:
#REMOVE EMPTY AND SHORT CONTENT
    if not docs:
        return []
    filtered = [
        doc for doc in docs
        if doc.page_content and len(doc.page_content.strip()) >= MIN_CONTENT_LENGTH
    ]
#REMOVE DUPLICATES
    seen = set()
    unique_docs = []

    for doc in filtered:
        content = doc.page_content.strip()

        if content not in seen:
            unique_docs.append(doc)
            seen.add(content)

    # 3️⃣ Limit documents
    return unique_docs[:MAX_DOCS]