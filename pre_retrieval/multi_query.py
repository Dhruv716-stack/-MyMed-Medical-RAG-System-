import os
from typing import List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Short generation task — Ollama is fine here
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.3,   # slight variation for diverse queries
    base_url="http://localhost:11434",
)

MULTI_QUERY_PROMPT = """
You are a medical search assistant.

Generate exactly 3 different search queries for the following question.
Each query should approach the topic from a slightly different angle
to improve document retrieval coverage.

Rules:
- One query per line
- No numbering, no bullets, no explanations
- Keep each query concise and medically precise
- Do NOT answer the question

Question: {query}

3 Search Queries:
"""

def generate_multi_queries(query: str) -> List[str]:
    prompt  = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    chain   = prompt | model | StrOutputParser()
    result  = chain.invoke({"query": query}).strip()

    # Parse — one query per line
    queries = [
        q.strip()
        for q in result.split("\n")
        if q.strip() and len(q.strip()) > 10
    ]

    # Safety: always return at least the original query
    if not queries:
        return [query]

    return queries[:3]