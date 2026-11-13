"""User-facing routes: chat page, RAG query, PDF report, history."""
import json
import logging
import os
import time

from flask import Blueprint, jsonify, render_template, request, send_file
from flask_login import current_user, login_required

from .. import db
from ..models import QueryLog

main_bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.post("/ask")
@login_required
def ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify(error="Please enter a question."), 400

    # Imported lazily: the embedding/LLM stack is heavy and must not delay
    # application startup (or block health checks on deploy).
    from ..services.rag import answer_ipc_query

    started = time.perf_counter()
    try:
        answer, sections = answer_ipc_query(query)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.exception("Answer generation failed for query %r", query)

        # Record the failure without masking the original problem.
        try:
            db.session.add(
                QueryLog(
                    user_id=current_user.id,
                    query_text=query,
                    status="FAILED",
                    error_message=f"{type(exc).__name__}: {exc}"[:2000],
                    latency_ms=latency_ms,
                )
            )
            db.session.commit()
        except Exception:  # noqa: BLE001 - never fail the error path
            db.session.rollback()

        return jsonify(error=_failure_hint(exc), detail=type(exc).__name__), 500

    latency_ms = (time.perf_counter() - started) * 1000
    log = QueryLog(
        user_id=current_user.id,
        query_text=query,
        response_text=answer,
        retrieved_sections=json.dumps(sections, ensure_ascii=False),
        status="SUCCESS",
        latency_ms=latency_ms,
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(
        answer=answer,
        sections=sections,
        latency_ms=round(latency_ms, 2),
        log_id=log.id,
    )


def _failure_hint(exc: Exception) -> str:
    """Translate an internal failure into something the operator can act on."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, FileNotFoundError) or "vector index" in text:
        return (
            "The legal knowledge base index is missing on the server. "
            "Build it with `python scripts/build_index.py --format numpy`."
        )
    if "all llm providers failed" in text:
        return (
            "No LLM provider could answer. Check that GROQ_API_KEY / "
            "MINIMAX_API_KEY / GEMINI_API_KEY are set and in quota."
        )
    if "no module named" in text:
        return "A required package is missing on the server; check the build log."
    return "Answer generation failed. Open /healthz for service status."


@main_bp.get("/report")
@login_required
def report():
    logs = (
        QueryLog.query.filter_by(user_id=current_user.id, status="SUCCESS")
        .order_by(QueryLog.created_at.desc())
        .limit(50)
        .all()
    )
    if not logs:
        return jsonify(error="No successful queries to export yet."), 404

    from ..services.rag import generate_report

    logs.reverse()
    path = generate_report(
        [entry.query_text for entry in logs],
        [entry.response_text or "" for entry in logs],
    )
    return send_file(
        os.path.abspath(path),
        as_attachment=True,
        download_name="AI_Lawyer_Report.pdf",
        mimetype="application/pdf",
    )


@main_bp.get("/api/history")
@login_required
def history():
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    pagination = (
        QueryLog.query.filter_by(user_id=current_user.id)
        .order_by(QueryLog.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify(
        items=[entry.to_dict() for entry in pagination.items],
        page=pagination.page,
        pages=pagination.pages,
        per_page=pagination.per_page,
        total=pagination.total,
    )
