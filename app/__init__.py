"""Application factory, extensions and shared configuration."""
import os
from pathlib import Path
from urllib.parse import quote_plus

from flask import Flask, jsonify, redirect, request, url_for
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()

BASE_DIR = Path(__file__).resolve().parent.parent


def _mysql_uri() -> str | None:
    """Build a MySQL URI from discrete MYSQL_* variables, if provided."""
    host = os.getenv("MYSQL_HOST")
    if not host:
        return None

    user = quote_plus(os.getenv("MYSQL_USER", "root"))
    password = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    port = os.getenv("MYSQL_PORT", "3306")
    name = os.getenv("MYSQL_DATABASE", "ipc_app")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


def _database_uri(app: Flask) -> str:
    """Resolve the SQLAlchemy URI.

    Priority: MYSQL_* variables > DATABASE_URL > local SQLite file.
    """
    mysql = _mysql_uri()
    if mysql:
        return mysql

    url = os.getenv("DATABASE_URL")
    if url:
        # Managed providers often hand out "mysql://..."; SQLAlchemy needs the
        # explicit DBAPI driver.
        if url.startswith("mysql://"):
            url = url.replace("mysql://", "mysql+pymysql://", 1)
        return url

    return f"sqlite:///{Path(app.instance_path) / 'ipc_app.db'}"


def create_app(config: dict | None = None) -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key-change-me"),
        SQLALCHEMY_DATABASE_URI=_database_uri(app),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # Keeps pooled MySQL connections alive across idle periods / restarts.
        SQLALCHEMY_ENGINE_OPTIONS={
            "pool_pre_ping": True,
            "pool_recycle": 280,
        },
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login_page"

    from .models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    @login_manager.unauthorized_handler
    def unauthorized():
        """Return JSON for API/XHR callers, redirect browsers to the login page."""
        wants_json = (
            request.path.startswith("/api/")
            or request.path in {"/ask", "/report"}
            or request.is_json
            or "application/json" in (request.headers.get("Accept") or "")
        )
        if wants_json:
            return jsonify(error="Authentication required."), 401
        return redirect(url_for("auth.login_page"))

    from .admin.routes import admin_bp
    from .auth.routes import auth_bp
    from .main.routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    return app
