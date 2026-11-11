"""Authentication routes: register, login, logout, profile."""
import re

from flask import Blueprint, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .. import db
from ..models import User

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _payload() -> dict:
    """Accept JSON or classic form submissions."""
    return request.get_json(silent=True) or request.form or {}


@auth_bp.get("/login")
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    return render_template("login.html")


@auth_bp.get("/register")
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))
    return render_template("register.html")


@auth_bp.post("/register")
def register():
    data = _payload()
    email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email):
        return jsonify(error="Please provide a valid email address."), 400
    if len(username) < 3:
        return jsonify(error="Username must be at least 3 characters."), 400
    if len(password) < MIN_PASSWORD_LENGTH:
        return jsonify(error=f"Password must be at least {MIN_PASSWORD_LENGTH} characters."), 400
    if User.query.filter_by(email=email).first():
        return jsonify(error="An account with that email already exists."), 409
    if User.query.filter_by(username=username).first():
        return jsonify(error="That username is already taken."), 409

    user = User(email=email, username=username, role="user")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    return jsonify(user=user.to_dict()), 201


@auth_bp.post("/login")
def login():
    data = _payload()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify(error="Invalid email or password."), 401
    if not user.is_active:
        return jsonify(error="This account has been deactivated."), 403

    login_user(user, remember=bool(data.get("remember")))
    return jsonify(user=user.to_dict())


@auth_bp.get("/logout")
@login_required
def logout():
    logout_user()
    if request.args.get("json") == "1":
        return jsonify(message="Logged out.")
    return redirect(url_for("auth.login_page"))


@auth_bp.get("/api/me")
@login_required
def me():
    return jsonify(user=current_user.to_dict(include_counts=True))
