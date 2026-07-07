import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# YES/NO task — qwen2.5:3b handles this perfectly, no Groq needed
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.0,   # deterministic for classification
    base_url="http://localhost:11434",
)

AMBIGUITY_PROMPT = """
You are a medical query analyzer.

Determine if the following query is ambiguous.

A query is ambiguous if:
- It lacks medical context
- It uses vague references (e.g. "this condition", "the medication")
- It cannot be clearly understood for document retrieval

Return ONLY one word: YES or NO

Query: {query}
"""

def is_ambiguous_llm(query: str) -> bool:
    prompt  = ChatPromptTemplate.from_template(AMBIGUITY_PROMPT)
    chain   = prompt | model | StrOutputParser()
    result  = chain.invoke({"query": query}).strip().upper()

    # Safety: if response is unexpected, default to not ambiguous
    return "YES" in result