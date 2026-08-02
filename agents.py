from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


def clean(text: str) -> str:
    """Remove unwanted Markdown formatting."""
    return (
        text.replace("**", "")
            .replace("###", "")
            .replace("##", "")
            .replace("#", "")
            .strip()
    )


def summarize_tasks(state: dict) -> dict:
    all_tasks = "\n".join(state["tasks"]) if isinstance(state["tasks"], list) else state["tasks"]

    prompt = f"""
You are an AI assistant that summarizes task lists.

Write exactly 2 sentences.

Rules:
- Do not list or rewrite individual tasks.
- Group tasks into general categories (work, study, personal).
- Mention urgent deadlines only if they exist.
- Give a high-level overview.

Task List:
{all_tasks}
"""

    result = llm.invoke(prompt)

    return {
        "summary": clean(result.content)
    }


def classify_tasks(state: dict) -> dict:
    all_tasks = "\n".join(state["tasks"]) if isinstance(state["tasks"], list) else state["tasks"]

    prompt = f"""
Classify each task into one of these categories:

- Work
- Study
- Personal

Task List:
{all_tasks}

Rules:
- Return plain text only.
- Do not use Markdown.
- Group the tasks under their category headings.

Example:

Work:
- Finish report

Study:
- Read LangGraph documentation

Personal:
- Buy groceries
"""

    result = llm.invoke(prompt)

    return {
        "classification": clean(result.content)
    }


def prioritize_tasks(state: dict) -> dict:
    all_tasks = "\n".join(state["tasks"]) if isinstance(state["tasks"], list) else state["tasks"]

    prompt = f"""
Prioritize the following tasks.

Task List:
{all_tasks}

Assign each task one priority:

High
Medium
Low

Rules:
- Return plain text only.
- No Markdown.
- Give one short reason for each task.

Example:

High:
- Finish AI assignment: Due tomorrow.

Medium:
- Study LangGraph: Important for learning.

Low:
- Call Mom: No urgent deadline.
"""

    result = llm.invoke(prompt)

    return {
        "priority": clean(result.content)
    }


def create_plan(state: dict) -> dict:
    prompt = f"""
Create the best execution plan using the information below.

Summary:
{state["summary"]}

Classification:
{state["classification"]}

Priority:
{state["priority"]}

Rules:
- Return plain text only.
- Number the steps.
- Keep each step short.
- Start with the highest-priority tasks.

Example:

1. Finish AI assignment
2. Study LangGraph
3. Buy groceries
4. Call Mom

End with one short sentence explaining why this order is recommended.
"""

    result = llm.invoke(prompt)

    return {
        "plan": clean(result.content)
    }