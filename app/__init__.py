"""Application factory, extensions and shared configuration."""
import logging
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


def _prepare_database(app: Flask) -> None:
    """Create tables and make sure an admin account exists (idempotent).

    This runs on startup so a dashboard-managed service works even when
    ``preDeployCommand`` (which only applies to blueprint deploys) never ran.
    Failures are logged instead of raised, so a bad DSN shows up as a clear
    error in the logs rather than a silently dead container.
    """
    from .models import User

    try:
        with app.app_context():
            db.create_all()

            email = (os.getenv("ADMIN_EMAIL") or "admin@local.dev").strip().lower()
            admin = User.query.filter_by(email=email).first()

            if admin is None:
                admin = User(
                    email=email,
                    username=os.getenv("ADMIN_USERNAME", "admin"),
                    role="admin",
                    is_active=True,
                )
                admin.set_password(os.getenv("ADMIN_PASSWORD", "admin12345"))
                db.session.add(admin)
                db.session.commit()
                app.logger.info("Seeded initial admin account: %s", email)
            else:
                # If ADMIN_PASSWORD is set explicitly it is the source of truth,
                # so a password that drifted from the env value is corrected.
                wanted = os.getenv("ADMIN_PASSWORD")
                if wanted and not admin.check_password(wanted):
                    admin.set_password(wanted)
                    db.session.commit()
                    app.logger.warning("Admin password re-synced from ADMIN_PASSWORD: %s", email)
    except Exception:  # noqa: BLE001 - never block startup on DB issues
        db.session.rollback()
        app.logger.exception(
            "Database bootstrap failed. Check MYSQL_*/DATABASE_URL and that the "
            "database accepts remote connections."
        )


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

    @app.teardown_appcontext
    def _clean_session(exception=None):
        """Never leave a poisoned session behind.

        A single failed flush (missing table, duplicate row, lost MySQL
        connection) otherwise poisons every later request with SQLAlchemy's
        "this Session's transaction has been rolled back" (e3q8) error.
        """
        if exception is not None:
            db.session.rollback()
        db.session.remove()

    from .admin.routes import admin_bp
    from .auth.routes import auth_bp
    from .main.routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    @app.get("/healthz")
    def healthz():
        """Diagnostics: database, vector index and provider key presence."""
        from sqlalchemy import text

        index_path = BASE_DIR / "vectorstore" / "ipc_index.npz"
        status = {
            "status": "ok",
            "database": "ok",
            "vector_index": "ok" if index_path.exists() else "missing",
            "vector_backend": os.getenv("VECTOR_BACKEND", "numpy"),
            "embedding_backend": os.getenv("EMBEDDING_BACKEND", "hf"),
            "providers": {
                "groq": bool(os.getenv("GROQ_API_KEY")),
                "minimax": bool(os.getenv("MINIMAX_API_KEY")),
                "gemini": bool(os.getenv("GEMINI_API_KEY")),
            },
        }
        code = 200

        try:
            db.session.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            status["database"] = f"error: {exc.__class__.__name__}"
            code = 503

        if status["vector_index"] == "missing":
            code = 503
        if not any(status["providers"].values()):
            status["status"] = "degraded (no LLM API key configured)"
            code = 503

        return jsonify(status), code

    # Tests create and seed their own database.
    if not app.config.get("TESTING"):
        _prepare_database(app)

    return app
