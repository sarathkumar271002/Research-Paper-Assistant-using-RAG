import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def groq_chat(model_name: str, temperature: float = 0.2, api_key: str = None):
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(model=model_name, temperature=temperature)
