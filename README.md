
# Multi-Task AI Web Application

A Flask-based AI web application demonstrating 4 core LLM architectures using LangChain and LangGraph.

## Features
- **Task 1: Stateful Task Planner** — Persistent conversation memory using LangGraph checkpointers.
- **Task 2: AI Tool Calling** — Automated tool invocation for real-time datetime retrieval and math calculations.
- **Task 3: RAG Document Q&A** — Upload PDF/DOCX files into ChromaDB for grounded context-restricted Q&A.
- **Task 4: Human-in-the-Loop Decision Making** — Interactive approval workflow for AI-generated plans.

## Quickstart
1. Clone the repo: `git clone <repo-url>`
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set `.env`: `OPENAI_API_KEY=your_key_here`
5. Run application: `python app.py`