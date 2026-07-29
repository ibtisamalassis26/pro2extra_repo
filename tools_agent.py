from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools import get_current_datetime, calculator

load_dotenv()

# Bind available tools to the LLM model
tools = [get_current_datetime, calculator]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def run_tool_agent(user_prompt: str) -> str:
    """Executes a reasoning loop with mandatory tool usage for calculations and dates."""
    messages = [
        SystemMessage(
            content=(
                "You are a strict reasoning AI agent with tools. "
                "Follow these rules:\n"
                "1. If you need today's date, MUST call `get_current_datetime` first.\n"
                "2. If you need to subtract dates or calculate numbers, MUST call `calculator` with the exact math expression.\n"
                "3. Do NOT perform math in your head. Always delegate math to the `calculator` tool.\n"
                "4. Synthesize the final answer using the outputs from the tools."
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    # Max iteration count to prevent infinite loops
    for _ in range(5):
        response = llm.invoke(messages)
        messages.append(response)

        # If no tool calls requested, we have our final answer
        if not response.tool_calls:
            return response.content

        # Execute all requested tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tools_by_name:
                selected_tool = tools_by_name[tool_name]
                tool_output = selected_tool.invoke(tool_args)

                print(f"[DEBUG TOOL EXECUTED] {tool_name}({tool_args}) -> {tool_output}")

                messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )

    return response.content