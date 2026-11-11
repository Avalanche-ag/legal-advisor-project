"""User-facing routes: chat page, RAG query, PDF report, history."""
import json
import os
import time

from flask import Blueprint, jsonify, render_template, request, send_file
from flask_login import current_user, login_required

from .. import db
from ..models import QueryLog

main_bp = Blueprint("main", __name__)


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
        db.session.add(
            QueryLog(
                user_id=current_user.id,
                query_text=query,
                status="FAILED",
                error_message=str(exc),
                latency_ms=latency_ms,
            )
        )
        db.session.commit()
        return jsonify(error="Generation failed. Please try again."), 500

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
