from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import torch
load_dotenv()

embedding=HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

