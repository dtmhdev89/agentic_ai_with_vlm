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
from src.chains.image_describer import ImageDescriber
from src.agent_tools.image_describer_tool import ImageDescriberTool
from src.models.yolo import Yolo
from src.agent_tools.object_detection_tool import ObjectDetectionTool
from src.agents.vision_agent import VisionAgent


if __name__ == "__main__":
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=1000
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

    # research_agent = ResearchAgent(
    #     model_name="llama3.1:8b",
    #     llm_mode="local_ollama",
    #     tools=[arxiv, wikipedia]
    # )
    # research_agent.build_agent_with_instruction(
    #     instruction_prompt=research_prompt
    # )

    # for chunk in research_agent.agent.stream(
    #     {"messages": [{"role": "user", "content": "machine learning"}]}
    # ):
    #     Messages.pretty_print_messages(chunk)

    # image_describer_agent = ImageDescriber(llm_mode='ollama')
    # results = image_describer_agent.chain.invoke({
    #     "image_path_or_url": "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"}
    # )

    # print(results)

    # image_describer_tool = ImageDescriberTool()
    # results = image_describer_tool.invoke("https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png")
    # print(results)

    # yolo_model = Yolo()
    # results = yolo_model.predict(
    #     "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg",
    #     verbose=False
    # )

    # results[0].show()

    # results = ObjectDetectionTool.detect_and_count_objects.invoke(
    #     "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg"
    # )

    # print(results)

    vision_prompt = (
        "You are a vision agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with visual tasks (e.g., describing images, detecting and counting objects)\n"
        "- Use only the tools provided to analyze visual inputs\n"
        "- After completing your task, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    vision_agent = VisionAgent(
        model_name='llama3.1:8b',
        llm_mode='ollama',
        prompt=vision_prompt,
        tools=[ImageDescriberTool(), ObjectDetectionTool.detect_and_count_objects]
    )

    for chunk in vision_agent.stream(
        {"messages": [{"role": "user", "content": "how many dog in image: https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg"}]}
    ):
        Messages.pretty_print_messages(chunk)
