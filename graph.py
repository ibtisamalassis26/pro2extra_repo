from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator
from agents import (
    summarize_tasks,
    classify_tasks,
    prioritize_tasks,
    create_plan,
)

class TaskState(TypedDict):
    # Appends new task submissions to previous ones instead of overwriting
    tasks: Annotated[List[str], operator.add]
    summary: str
    classification: str
    priority: str
    plan: str


builder = StateGraph(TaskState)

builder.add_node("summarizer", summarize_tasks)
builder.add_node("classifier", classify_tasks)
builder.add_node("prioritizer", prioritize_tasks)
builder.add_node("planner", create_plan)

builder.add_edge(START, "summarizer")
builder.add_edge(START, "classifier")
builder.add_edge(START, "prioritizer")

builder.add_edge("summarizer", "planner")
builder.add_edge("classifier", "planner")
builder.add_edge("prioritizer", "planner")

builder.add_edge("planner", END)

memory = MemorySaver()
task_graph = builder.compile(checkpointer=memory)