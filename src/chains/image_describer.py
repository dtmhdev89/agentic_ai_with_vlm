from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from typing import Literal
from src.LLM_chat.ollama import Ollama
from src.LLM_chat.openai import Openai
from src.utils.image import Image as ImageUtils


class ImageDescription(BaseModel):
    image_description: str = Field(
        description="Detailed description of the image"
    )


class ImageDescriber:
    """Image Describer chain"""

    def __init__(
        self,
        llm_mode: Literal['ollama', 'openai']
    ) -> None:
        self._llm_mode = llm_mode
        self._llm = self._init_llm()
        self._chain = self._init_chain()

    @property
    def chain(self):
        """chain property"""
        return self._chain

    def _init_llm(self):
        """Chat LLMs init"""
        llm = ""
        if self._llm_mode == 'ollama':
            llm = Ollama(model_name="llama3.1:8b")
        elif self._llm_mode == "openai":
            llm = Openai(model_name="gpt-4.1-nano")

        return llm

    def _image_describer_prompt_func(self, inputs: dict):
        """Prompt Creator"""

        image_path_or_url = inputs["image_path_or_url"]
        image_b64, image_mime_type = ImageUtils.encode_image(
            image_path_or_url,
            get_mime_type=True
        )

        image_describer_chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="""You are an expert image describer. When presented with an image, provide a detailed, accurate, and objective description of its visible content. Focus on aspects such as:
                - Objects present, their positions, and relationships
                - Colors, lighting, composition, and textures
                - Actions or dynamics, if any (e.g., people walking, water flowing)
                - Contextual or inferred information (e.g., likely setting, era, or activity)

                Avoid adding information that is not visible or cannot be reasonably inferred from the image. Do not speculate or inject personal opinion unless explicitly requested. If text appears in the image, transcribe it accurately."""),
            HumanMessage(content=[
                {"type": "text", "text": "Describe the following image for me:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}", "detail": "low"}
                }
            ])
        ])

        return image_describer_chat_template.invoke({})

    def _init_chain(self):
        chain = (
            self._image_describer_prompt_func
            | self._llm.with_structured_output(ImageDescription)
        )
        return chain
