import io
import os

from flask import Flask, jsonify, render_template, request, send_file

from rag_pipeline import answer_ipc_query, generate_report

app = Flask(__name__)

# Keep the last conversation in memory for the PDF export button.
# Fine for a single-user demo; swap for a DB/session store if you scale.
_last_query: list[str] = []
_last_answer: list[str] = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify(error="Please enter a question."), 400

    try:
        answer, sections = answer_ipc_query(query)
    except Exception as exc:
        return jsonify(error=f"Generation failed: {exc}"), 502

    _last_query.append(query)
    _last_answer.append(answer)
    # Cap the in-memory transcript so it can't grow forever.
    del _last_query[:-50]
    del _last_answer[:-50]

    return jsonify(answer=answer, sections=sections)


@app.route("/report")
def report():
    if not _last_query:
        return jsonify(error="No conversation to export yet."), 400
    path = generate_report(list(_last_query), list(_last_answer))
    return send_file(
        os.path.abspath(path),
        as_attachment=True,
        download_name="AI_Lawyer_Report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))