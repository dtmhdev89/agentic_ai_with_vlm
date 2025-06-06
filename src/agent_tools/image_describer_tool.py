from langchain_core.tools import BaseTool
from typing import Optional
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
from src.chains.url_extractor import UrlExtractor
from src.chains.image_describer import ImageDescriber


class ImageDescriberInput(BaseModel):
    text: str = Field(description="Path or URL to the image in the format PNG or JPG/JPEG")


class ImageInput(BaseModel):
    image_path_or_url: str = Field(description="Image path or URL")


class ImageDescriberTool(BaseTool):
    """Image Describer Tool
    This class is a custom LangChain tool designed to be used by an AI agent (e.g., a LangChain agent).
    When the agent decides it needs to describe an image,
    it will "call" this tool, providing the image's location as input.
    """

    # A unique string identifier for the tool. This is how the LLM agent refers to the tool.
    name: str = "image_describer"
    # A clear explanation of what the tool does. Again, the LLM reads this to decide when to use the tool.
    description: str = "This tool can describe the image in a detailed way"
    # This links the tool to its input schema (ImageDescriberInput).
    # LangChain uses this to generate the correct JSON schema for the LLM's function calling mechanism.
    args_schema: Optional[ArgsSchema] = ImageDescriberInput
    # This is an important flag. If True, the tool's output is returned directly to the user
    # (or the next step of the LangChain sequence) without being
    # passed back to the LLM for further thought or processing
    return_direct: bool = True

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        extractor_chain = UrlExtractor(
            model_name="llama3.1:8b ",
            llm_mode='ollama'
        ).extractor_chain
        image_describer_agent = ImageDescriber(llm_mode='ollama').chain
        try:
            parsed: ImageInput = extractor_chain.invoke({"input": text})
        except Exception as e:
            return f"Failed to extract image URL: {str(e)}"

        image_path_or_url = parsed.image_path_or_url
        if not image_path_or_url:
            return "No image URL found in the input."

        output = image_describer_agent.invoke(
            {"image_path_or_url": image_path_or_url}
        )
        
        return output.image_description

    async def _arun(
        self,
        image_path_or_url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            image_path_or_url,
            run_manager=run_manager
        )
