"""Admin dashboard page and management/metrics APIs."""
from flask import Blueprint, jsonify, render_template, request
from sqlalchemy import func

from .. import db
from ..auth.decorators import admin_required
from ..models import QueryLog, User

admin_bp = Blueprint("admin", __name__)


@admin_bp.get("/admin/dashboard")
@admin_required
def dashboard():
    return render_template("admin.html")


@admin_bp.get("/api/admin/users")
@admin_required
def list_users():
    rows = (
        db.session.query(User, func.count(QueryLog.id))
        .outerjoin(QueryLog, QueryLog.user_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at.asc())
        .all()
    )
    return jsonify(
        users=[
            {**user.to_dict(), "query_count": int(count or 0)}
            for user, count in rows
        ]
    )


@admin_bp.patch("/api/admin/users/<int:user_id>/role")
@admin_required
def update_role(user_id: int):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify(error="User not found."), 404

    role = ((request.get_json(silent=True) or {}).get("role") or "").strip().lower()
    if role not in ("admin", "user"):
        return jsonify(error="Role must be 'admin' or 'user'."), 400

    user.role = role
    db.session.commit()
    return jsonify(user=user.to_dict())


@admin_bp.patch("/api/admin/users/<int:user_id>/toggle-status")
@admin_required
def toggle_status(user_id: int):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify(error="User not found."), 404

    from flask_login import current_user

    if user.id == current_user.id:
        return jsonify(error="You cannot deactivate your own account."), 400

    user.is_active = not user.is_active
    db.session.commit()
    return jsonify(user=user.to_dict())


@admin_bp.get("/api/admin/logs")
@admin_required
def list_logs():
    status = (request.args.get("status") or "").upper()
    sort = (request.args.get("sort") or "date").lower()
    limit = min(request.args.get("limit", 100, type=int), 500)

    query = QueryLog.query
    if status in ("SUCCESS", "FAILED"):
        query = query.filter(QueryLog.status == status)
    if sort == "latency":
        query = query.order_by(QueryLog.latency_ms.desc())
    else:
        query = query.order_by(QueryLog.created_at.desc())

    return jsonify(logs=[entry.to_dict() for entry in query.limit(limit).all()])


@admin_bp.get("/api/admin/stats")
@admin_required
def stats():
    total_users = db.session.query(func.count(User.id)).scalar() or 0
    total_queries = db.session.query(func.count(QueryLog.id)).scalar() or 0
    success_count = (
        db.session.query(func.count(QueryLog.id))
        .filter(QueryLog.status == "SUCCESS")
        .scalar()
        or 0
    )
    avg_latency = (
        db.session.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0
    )

    return jsonify(
        total_users=int(total_users),
        total_queries=int(total_queries),
        success_rate=round((success_count / total_queries) * 100, 2) if total_queries else 0.0,
        avg_latency_ms=round(float(avg_latency), 2),
    )
