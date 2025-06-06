from typing import Literal, List
from langgraph.prebuilt import create_react_agent
from src.LLM_chat.ollama import Ollama
from src.LLM_chat.openai import Openai


class VisionAgent:
    """Vision Agent"""

    def __new__(
        cls,
        model_name: Literal['llama3.1:8b', 'qwen3:8b'],
        llm_mode: Literal['ollama', 'openai'],
        prompt: str,
        tools: List
    ):
        return cls._create_agent(
            model_name,
            llm_mode,
            prompt,
            tools
        )
        
    @classmethod
    def _create_agent(
        cls,
        model_name: str,
        llm_mode: Literal['ollama', 'openai'],
        prompt: str,
        tools: List
    ):
        llm = ''
        if llm_mode == 'ollama':
            llm = Ollama(model_name=model_name)
        elif llm_mode == 'openai':
            llm = Openai(model_name=model_name)
        
        return create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
            name="vision_agent"
        )
        
