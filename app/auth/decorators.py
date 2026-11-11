"""Authentication helpers and route protection decorators."""
from functools import wraps

from flask import abort, jsonify, request
from flask_login import current_user, login_required


def admin_required(view):
    """Allow only authenticated users whose role is 'admin'."""

    @login_required
    @wraps(view)
    def decorated(*args, **kwargs):
        if getattr(current_user, "role", None) != "admin":
            if request.path.startswith("/api/"):
                return jsonify(error="Admin privileges required."), 403
            abort(403)
        return view(*args, **kwargs)

    return decorated
