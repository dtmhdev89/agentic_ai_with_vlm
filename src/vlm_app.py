from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from src.agents.research_agent import ResearchAgent
from src.utils.messages import Messages


if __name__ == "__main__":
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=2, doc_content_chars_max=1000
    )
    arxiv = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        description="Search for papers on a given topic using Arxiv"
    )
    # results = arxiv.invoke("GraphRAG")
    # print(results)

    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia = WikipediaQueryRun(
        api_wrapper=wikipedia_wrapper,
        description="Search for information on a given topic using Wikipedia"
    )
    # results = wikipedia.invoke("Ho Chi Minh")
    # print(results)

    research_prompt = (
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    research_agent = ResearchAgent(
        model_name="llama3.1:8b",
        llm_mode="local_ollama",
        tools=[arxiv, wikipedia]
    )
    research_agent.build_agent_with_instruction(
        instruction_prompt=research_prompt
    )

    for chunk in research_agent.agent.stream(
        {"messages": [{"role": "user", "content": "machine learning"}]}
    ):
        Messages.pretty_print_messages(chunk)
