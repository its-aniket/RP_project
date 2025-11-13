"""
utils/llm_utils.py
------------------
Central utility for managing LLM instances across the project.
Supports Groq (Llama 3) by default.
"""

import os
from langchain_groq import ChatGroq  # pip install langchain_groq

# You can set GROQ_API_KEY in your environment:
#   setx GROQ_API_KEY "your_api_key_here"
# or for Linux/Mac:
#   export GROQ_API_KEY="your_api_key_here"


def get_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens: int = 6000,
):
    """
    Returns a configured LLM instance (Groq Chat model by default).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ GROQ_API_KEY not set. Please export it in your environment.")

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key,
    )

    return llm
