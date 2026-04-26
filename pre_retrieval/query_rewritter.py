
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from generation.retrieve_model import model


# Prompt template for rewriting
QUERY_REWRITE_PROMPT = """
You are a medical information retrieval assistant.

Your job is to convert a user's search query into a clear,
formal medical question that would help retrieve information
from medical textbooks or clinical documents.
Rules:
- Convert the query into a complete question.
- Convert the query into a complete question.
- Preserve the original medical terms.
- Expand abbreviations if obvious (bp → blood pressure).
- Do NOT introduce new diseases or concepts.
- Keep the meaning identical.
- Remove conversational language
- Make the query concise and retrieval-friendly
STRICT RULES:
- DO NOT introduce new medical terminology.
- DO NOT replace disease names.
- Keep the same key medical entities.
- Only clarify wording and remove conversational phrases.
- Keep the rewritten query close to the original.

If the query is already good, return it unchanged.

User question:
{query}

Rewritten search query:
"""


def rewrite_query(query: str) -> str:
    """
    Rewrite a query for better retrieval without changing meaning.
    """
    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)

    chain = prompt | model | StrOutputParser()

    rewritten_query = chain.invoke(
        {
            "query": query
        }
    )

    final_query=rewritten_query.strip()

    return final_query