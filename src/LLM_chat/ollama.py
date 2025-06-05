from langchain_ollama import ChatOllama
from typing import Literal
import os


class Ollama:
    """ChatOllama"""

    def __new__(
        cls,
        model_name: Literal['llama3.1:8b', 'qwen3:8b'],
        temperature=0
    ):
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_URI")
        )
