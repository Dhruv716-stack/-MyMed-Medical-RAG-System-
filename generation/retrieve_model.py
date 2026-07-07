# generation/retrieve_model.py

import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras

load_dotenv()

# v3 change: generation model moved from Groq llama-3.1-8b-instant
# to Cerebras gpt-oss-120b to improve answer faithfulness/grounding.
# Cerebras free tier caps gpt-oss-120b at 5 req/min, so max_retries gives
# automatic backoff on 429 (the eval loop also throttles between calls).
model = ChatCerebras(
    model="gpt-oss-120b",
    temperature=0.2,
    max_tokens=1500,  # v2 change (was 1024)
    max_retries=5,
    api_key=os.getenv("CEREBRAS_API_KEY"),
)
