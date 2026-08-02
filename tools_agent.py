from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools import (
    get_current_datetime, 
    calculate_future_or_past_date, 
    calculate_days_until, 
    calculator
)

load_dotenv()

tools = [
    get_current_datetime, 
    calculate_future_or_past_date, 
    calculate_days_until, 
    calculator
]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(tools)


def run_tool_agent(user_prompt: str) -> str:
    """Executes a reasoning loop with mandatory tool usage for calculations and dates."""
    messages = [
        SystemMessage(
            content=(
                "You are a strict reasoning AI agent with specialized tools.\n"
                "1. If asked about a date X days from now/past, MUST call `calculate_future_or_past_date(days)`.\n"
                "2. If asked how many days remain until a specific future date, MUST call `calculate_days_until(target_date_str)` using YYYY-MM-DD format.\n"
                "3. If you need the exact current timestamp, call `get_current_datetime`.\n"
                "4. If you need standard mathematical calculations, call `calculator`.\n"
                "5. Synthesize the final answer using tool responses."
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    for _ in range(5):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

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