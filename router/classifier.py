from generation.retrieve_model import model


def classify_query(
    query
):

    prompt = f"""

Classify:

MEDICAL_RAG
GENERAL_CHAT
MEMORY

Query:
{query}

Return one label only.
"""

    result = model.invoke(
        prompt
    )

    return result.content.strip()