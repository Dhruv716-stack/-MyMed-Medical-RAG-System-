from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1
    
)
model=ChatHuggingFace(llm=llm)