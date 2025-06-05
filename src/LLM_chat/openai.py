from langchain_openai import ChatOpenAI


class Openai:
    """ChatOpenAI"""

    def __new__(
        cls,
        model_name,
        temperature=0
    ):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
