from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import os


class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Image path or URL")


class UrlExtractor:
    """Url Extractor Chain"""

    def __init__(
        self,
        model_name,
        llm_mode: Literal['ollama', 'openai']
    ):
        self._model_name = model_name
        self._llm_mode = llm_mode
        self._parser = PydanticOutputParser(
            pydantic_object=ImageInput
        )

        llm = self._init_llm()

        prompt = PromptTemplate.from_template(
            "Extract the image path or URL from the following input:\n\n{input}\n\n{format_instructions}"
        ).partial(format_instructions=self._parser.get_format_instructions())

        self._extractor_chain = prompt | llm | self._parser

    @property
    def extractor_chain(self):
        """extractor_chain property"""
        return self._extractor_chain

    def _init_llm(self):
        """LLM initialization"""
        llm = ""
        if self._llm_mode == 'ollama':
            llm = ChatOllama(
                model=self._model_name,
                base_url=os.getenv("OLLAMA_URI"),
                temperature=0
            )
        elif self._llm_mode == 'openai':
            llm = ChatOpenAI(model="gpt-4.1-nano")

        return llm
