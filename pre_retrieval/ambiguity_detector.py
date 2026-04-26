AMBIGUITY_PROMPT = """
You are a medical query analyzer.

Determine if the following query is ambiguous.

A query is ambiguous if:
- It lacks context
- It uses vague references
- It cannot be clearly understood for retrieval

Return only:
YES or NO

Query:
{query}
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from generation.retrieve_model import model


prompt = PromptTemplate.from_template(AMBIGUITY_PROMPT)


def is_ambiguous_llm(query: str) -> bool:
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"query": query}).strip().lower()

    return "yes" in result