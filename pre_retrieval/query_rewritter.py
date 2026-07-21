import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Lightweight task — Ollama local, saves Groq TPM for generation only
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.1,
    base_url="http://localhost:11434",
)

QUERY_REWRITE_PROMPT = """
You are a medical query optimizer for a RAG system.

Rewrite the user's query to make it:
- Clear
- Specific  
- Medically accurate
- Optimized for semantic search

Rules:
- Do NOT change the intent
- Do NOT add extra information
- Do NOT answer the query
- Return ONLY the rewritten query, nothing else

User Query: {query}

Rewritten Query:
"""

def rewrite_query(query: str) -> str:
    prompt  = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain   = prompt | model | StrOutputParser()

    # Retry once to handle Ollama cold-start (model loading on first call)
    for attempt in range(2):
        try:
            result = chain.invoke({"query": query}).strip()
            if result and len(result) <= 500:
                return result
        except Exception:
            if attempt == 0:
                continue  # retry once
            raise

    return query