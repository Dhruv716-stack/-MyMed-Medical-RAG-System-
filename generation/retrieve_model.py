import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load .env files for optional overrides.
workspace_root = Path.cwd()
project_root = workspace_root / "-MyMed-Medical-RAG-System-"
for env_path in [workspace_root / ".env", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, override=True)

# Match the previous HF model with an Ollama equivalent.
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = ChatOllama(
    model=ollama_model,
    temperature=0.2,
    base_url=ollama_base_url,
)