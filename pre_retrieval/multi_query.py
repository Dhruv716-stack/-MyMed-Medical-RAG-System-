# retrieval/pre_retrieval/multi_query.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1,
    top_p=0.9
    
)
model=ChatHuggingFace(llm=llm)


prompt = PromptTemplate.from_template("""
You are a medical query optimizer for a RAG system.

Your task is to generate 3 to 5 alternative versions of the user query in a formal question type format that would be more effective for retrieving information from medical documents.
STRICTLY for improving document retrieval.

IMPORTANT RULES:
- Stay strictly within the scope of the original query
- Do NOT introduce new topics (e.g., causes, prevention, risk factors)
- Do NOT assume additional medical conditions
- Do NOT expand beyond what is likely present in the document
- Only rephrase or slightly expand wording for clarity
- Preserve the original meaning of the question.
- Do NOT introduce new diseases or medical concepts.
- Do NOT change the topic.

Goal:
- Improve semantic matching with the document
- Not to broaden the question

Original Query:
{query}

Output:
Provide 3 rewritten queries, each on a new line:

""")

def generate_multi_queries(query: str, max_queries: int = 5):
    chain = prompt | model | StrOutputParser()
    output = chain.invoke({"query": query})

    queries = [
        q.strip("- ").strip()
        for q in output.split("\n")
        if q.strip()
    ]

    queries.append(query)

    queries = list(set(queries))[:max_queries]

    return queries