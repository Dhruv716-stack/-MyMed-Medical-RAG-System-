# generation/retrieve_model.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0.2, 
    max_tokens=1500,  # v2 change (was 1024)
    api_key=os.getenv("GROQ_API_KEY"),
)