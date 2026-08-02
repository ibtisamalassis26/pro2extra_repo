import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def clean_markdown(text: str) -> str:
  """Removes markdown headers (#) and bold/italic stars (* or _) for clean plain text."""
  text = re.sub(r"#+\s*", "", text)
  text = re.sub(r"[\*_]{1,3}", "", text)
  return text.strip()



initial_plan_prompt = ChatPromptTemplate.from_template("""
You are an expert AI productivity assistant. Based on the following raw tasks provided by the user, create a structured, detailed, step-by-step actionable plan.

Rules:
1. Start your response with the exact line: 'Initial Task Plan'
2. For each task, prioritize it logically and break it down into 3-4 specific, actionable sub-steps with recommended time allocations or strategies.
3. Do NOT use Markdown header symbols (#) or bold/italic symbols (*).

User Tasks:
{tasks}

Detailed Actionable Plan:
""")


refine_plan_prompt = ChatPromptTemplate.from_template("""
You are an expert AI productivity assistant. The user wants to modify their current task plan.

Current Plan:
{current_plan}

User Feedback / Request:
{feedback}

Rules:
1. Start your response with the exact line: 'Updated Task Plan'
2. Incorporate the user's feedback, re-prioritize, and keep the detailed sub-step breakdown for each item.
3. Do NOT use Markdown header symbols (#) or bold/italic symbols (*).

Detailed Updated Plan:
""")


def generate_initial_plan(tasks: str) -> str:
  """Generates a detailed, expanded initial draft of the plan."""
  chain = initial_plan_prompt | llm
  response = chain.invoke({"tasks": tasks})
  return clean_markdown(response.content)


def refine_plan(current_plan: str, feedback: str) -> str:
  """Updates and expands the existing plan based on human feedback."""
  chain = refine_plan_prompt | llm
  response = chain.invoke({"current_plan": current_plan, "feedback": feedback})
  return clean_markdown(response.content)