from src.agents.research_agent import ResearchAgent
from src.agents.math_agent import MathAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.utils.messages import Messages

if __name__ == "__main__":
    reseach_prompt = (
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    math_prompt = (
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You have access to tools: add, multiply, divide.\n"
        "- Use tools whenever possible to compute answers.\n"
        "- Do NOT do math in your head; always use a tool.\n"
        "- Respond ONLY with the result, no explanation.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
    )

    supervisor_prompt = (
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    )

    math_agent = MathAgent(
        model_name="llama3.1:8b",
        llm_mode="local_ollama"
    )

    math_agent.build_agent_with_instruction(instruction_prompt=math_prompt)

    research_agent = ResearchAgent(
        model_name="llama3.1:8b",
        llm_mode="local_ollama"
    )

    research_agent.build_agent_with_instruction(
        instruction_prompt=reseach_prompt
    )

    supervisor_agent = SupervisorAgent(
        model_name="llama3.1:8b",
        llm_mode='local_ollama'
    )

    supervisor_agent.build_agent_with_instruction(
        instruction_prompt=supervisor_prompt,
        subordinate_agents=[research_agent.agent, math_agent.agent]
    )

    # for chunk in math_agent.agent.stream(
    #     {"messages": [{"role": "user", "content": "What's the result of (5 + 100) / 5 ?"}]}
    # ):
    #     Messages.pretty_print_messages(chunk)

    # for chunk in research_agent.agent.stream(
    #     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
    # ):
    #     Messages.pretty_print_messages(chunk)

    # chart_bytes = supervisor_agent.agent.get_graph().draw_mermaid_png()

    # with open('data_outputs/supervisor_agent_graph.png', 'wb') as writer:
    #     writer.write(chart_bytes)

    for chunk in supervisor_agent.agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Find GDP of Vietnam in 2023 and 2024, determine the percentage increase or decrease?",
                }
            ]
        },
    ):
        Messages.pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]
    print(final_message_history)
