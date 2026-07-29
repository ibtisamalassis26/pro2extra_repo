import os
from flask import Flask, render_template, request, session
import uuid
from werkzeug.utils import secure_filename
from graph import task_graph
from tools_agent import run_tool_agent
from rag_assistant import process_and_index_document, answer_rag_question
from human_in_loop import generate_initial_plan, refine_plan

app = Flask(__name__)
app.secret_key = "super-secret-key"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    # Task 4 session state initialization
    if "current_plan" not in session:
        session["current_plan"] = None
    if "plan_status" not in session:
        session["plan_status"] = None

    plan_result = None
    tool_result = None
    rag_status = None
    rag_answer = None

    if request.method == "POST":
        action = request.form.get("action")

        # Task 1: Task Planner with Memory
        if action == "plan_tasks":
            tasks_input = request.form.get("tasks", "")
            if tasks_input.strip():
                config = {"configurable": {"thread_id": session["thread_id"]}}
                plan_result = task_graph.invoke(
                    {"tasks": [tasks_input]}, config=config
                )

        # Task 2: AI Tool Assistant
        elif action == "query_tool":
            tool_query = request.form.get("tool_query", "")
            if tool_query.strip():
                tool_result = run_tool_agent(tool_query)

        # Task 3: Document Upload (RAG)
        elif action == "upload_doc":
            if "file" in request.files:
                file = request.files["file"]
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    rag_status = process_and_index_document(file_path)

        # Task 3: Ask Document Question (RAG)
        elif action == "query_rag":
            rag_query = request.form.get("rag_query", "")
            if rag_query.strip():
                rag_answer = answer_rag_question(rag_query)

        # Task 4: Human-in-the-Loop - Generate Draft Plan
        elif action == "create_draft_plan":
            draft_tasks = request.form.get("draft_tasks", "")
            if draft_tasks.strip():
                session["current_plan"] = generate_initial_plan(draft_tasks)
                session["plan_status"] = "in_review"

        # Task 4: Human-in-the-Loop - Modify Plan with Feedback
        elif action == "refine_plan":
            feedback = request.form.get("feedback", "")
            if feedback.strip() and session.get("current_plan"):
                session["current_plan"] = refine_plan(session["current_plan"], feedback)
                session["plan_status"] = "in_review"

        # Task 4: Human-in-the-Loop - Approve Plan
        elif action == "approve_plan":
            session["plan_status"] = "approved"

        # Task 4: Reset Plan State
        elif action == "reset_plan":
            session["current_plan"] = None
            session["plan_status"] = None

    return render_template(
        "index.html",
        result=plan_result,
        tool_result=tool_result,
        rag_status=rag_status,
        rag_answer=rag_answer,
        current_plan=session.get("current_plan"),
        plan_status=session.get("plan_status")
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)