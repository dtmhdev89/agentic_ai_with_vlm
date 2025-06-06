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
from src.agents.supervisor_agent import SupervisorAgent


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

    research_agent = ResearchAgent(
        model_name="llama3.1:8b",
        llm_mode="local_ollama",
        tools=[arxiv, wikipedia]
    )
    research_agent.build_agent_with_instruction(
        instruction_prompt=research_prompt
    )

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

    # for chunk in vision_agent.stream(
    #     {"messages": [{"role": "user", "content": "how many dog in image: https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg"}]}
    # ):
    #     Messages.pretty_print_messages(chunk)

    # supervisor_prompt = (
    #     "You are a supervisor managing two agents:\n"
    #     "- a research agent. Assign research-related tasks to this agent\n"
    #     "- a vision agent. Assist visual tasks (e.g., describing images, detecting and counting objects)\n"
    #     "Assign work to one agent at a time, do not call agents in parallel.\n"
    #     "Do not do any work yourself."
    # )

    supervisor_prompt = (
        "You are a supervisor managing two agents:\n"
        "- research_agent: Use this agent ONLY for research-related tasks (e.g., searching information, finding papers, summarizing documents).\n"
        "- vision_agent: Use this agent ONLY for visual tasks (e.g., describing images, detecting and counting objects).\n\n"
        "RULES:\n"
        "- Assign tasks to only one agent at a time.\n"
        "- Do NOT call multiple agents in parallel.\n"
        "- Do NOT perform any work yourself — always delegate.\n"
        "- Be concise when handing off tasks to agents, only provide what is necessary for them to complete the job."
    )

    supervisor_agent = SupervisorAgent(
        model_name="llama3.1:8b",
        llm_mode='ollama'
    ).build_agent_with_instruction(
        instruction_prompt=supervisor_prompt,
        subordinate_agents=[research_agent.agent, vision_agent]
    )

    for chunk in supervisor_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the latest research on positional embeddings?",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    print('---------------------')
    print(chunk["supervisor"]["messages"])

    for chunk in supervisor_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the concept visualized in the image? Image: https://huggingface.co/datasets/tmnam20/Storage/resolve/main/rope.png Provide me detailed information about the concept. If possible, give me some research papers about it.",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    print('---------------------')
    print(chunk["supervisor"]["messages"])

    for chunk in supervisor_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How many dogs are there in the image? Image: https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    print('---------------------')
    print(chunk["supervisor"]["messages"])


    for chunk in supervisor_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How many dogs are there in the image? Image: https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    print('--------------------------')
    print(chunk["supervisor"]["messages"])


    for chunk in supervisor_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What color fur does the dog in the the picture? and looking for more information about this dog? Image: https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/04/24144817/American-Staffordshire-Terrier-lying-outdoors-next-to-a-kitten-that-is-playing-with-the-dogs-nose.jpg",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    print(chunk["supervisor"]["messages"])
