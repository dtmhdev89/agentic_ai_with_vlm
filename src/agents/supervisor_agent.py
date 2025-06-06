from typing import Literal, List
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
import os


class SupervisorAgent:
    """Supervisor Agent"""

    def __init__(
        self,
        model_name: str,
        llm_mode: Literal['ollama', 'openai']
    ):
        self._model_name = model_name
        self._llm_mode = llm_mode
        self._agent: object = None

    @property
    def agent(self):
        """agent property"""
        return self._agent
    
    def _config_for_init_chat_model(self):
        base_configs = {
            "model": self._model_name,
            "model_provider": self._llm_mode
        }
        if self._llm_mode == 'ollama':
            base_configs.update(
                {"base_url": os.getenv("OLLAMA_URI")}
            )
        
        return base_configs

    def build_agent_with_instruction(
        self,
        instruction_prompt,
        subordinate_agents: List[object]
    ):
        """Build agent"""

        configs = self._config_for_init_chat_model()
        
        self._agent = create_supervisor(
            model=init_chat_model(
                **configs
            ),
            agents=subordinate_agents,
            prompt=instruction_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()

        return self._agent
