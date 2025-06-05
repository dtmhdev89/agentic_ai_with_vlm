from typing import Literal, List
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama


class SupervisorAgent:
    """Supervisor Agent"""

    def __init__(
        self,
        model_name: str,
        llm_mode: Literal['local_ollama']
    ):
        self._model_name = model_name
        self._llm_mode = llm_mode
        self._agent: object = None

    @property
    def agent(self):
        """agent property"""
        return self._agent
    
    def _init_llm_from_ollama_server(self):
        llm = ChatOllama(
            model=self._model_name,
            base_url="http://127.0.0.1:11434",
            streaming=True
        )

        return llm

    def build_agent_with_instruction(
        self,
        instruction_prompt,
        subordinate_agents: List[object]
    ):
        """Build agent"""

        self._agent = create_supervisor(
            model=init_chat_model(
                model=self._model_name,
                model_provider='ollama'
            ),
            agents=subordinate_agents,
            prompt=instruction_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()

        return self._agent
