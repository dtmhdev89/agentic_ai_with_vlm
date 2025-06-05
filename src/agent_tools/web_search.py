from langchain_tavily import TavilySearch


class WebSearch(TavilySearch):
    """Inherit from TavilySearch"""

    def __init__(self, max_results) -> None:
        super().__init__(max_results=max_results)
