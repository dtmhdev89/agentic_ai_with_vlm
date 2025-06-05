from langgraph.prebuilt import create_react_agent
from src.agent_tools.web_search import WebSearch
from langchain_ollama import ChatOllama


class ResearchAgent:
    """Research Agent has abilities of  ReAct Agent"""

    def __init__(
        self,
        model_name,
        llm_mode="local_ollama",
        tools=None
    ) -> None:
        self._llm_mode = llm_mode
        self._model_name = model_name
        self._agent_name = "research_agent"
        if tools:
            self._tools = tools
        else:
            self._tools = [WebSearch(max_results=3)]

        self._agent = None

    @property
    def agent(self):
        """agent getter"""

        return self._agent

    def _init_llm_from_ollama_server(self):
        llm = ChatOllama(
            model=self._model_name,
            base_url="http://127.0.0.1:11434",
            streaming=True
        )

        return llm

    def build_agent_with_instruction(self, instruction_prompt):
        """Build agent"""

        model = ""
        if self._llm_mode == "local_ollama":
            model = self._init_llm_from_ollama_server()
        else:
            model = self._model_name

        self._agent = create_react_agent(
            model=model,
            tools=self._tools,
            prompt=instruction_prompt,
            name=self._agent_name
        )

        return self._agent
