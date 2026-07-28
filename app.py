from flask import Flask, render_template, request, session
import uuid
from graph import task_graph

app = Flask(__name__)
app.secret_key = "super-secret-key"

@app.route("/", methods=["GET", "POST"])
def home():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    result = None
    if request.method == "POST":
        tasks_input = request.form.get("tasks", "")
        if tasks_input.strip():
            config = {"configurable": {"thread_id": session["thread_id"]}}
            # Pass input inside a list so operator.add combines past inputs
            result = task_graph.invoke({"tasks": [tasks_input]}, config=config)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    