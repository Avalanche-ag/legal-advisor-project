"""Relational models: users and persistent query logs."""
import json
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from . import db

ROLES = ("admin", "user")


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    queries = db.relationship(
        "QueryLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self, include_counts: bool = False) -> dict:
        data = {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_counts:
            data["query_count"] = self.queries.count()
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User {self.username} ({self.role})>"


class QueryLog(db.Model):
    __tablename__ = "query_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text = db.Column(db.Text, nullable=False)
    response_text = db.Column(db.Text, nullable=True)
    # JSON-encoded list of retrieved section dicts (citations).
    retrieved_sections = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False, index=True)
    error_message = db.Column(db.Text, nullable=True)
    latency_ms = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    user = db.relationship("User", back_populates="queries")

    def sections(self) -> list:
        """Decode the stored citation payload into a list."""
        if not self.retrieved_sections:
            return []
        try:
            value = json.loads(self.retrieved_sections)
            return value if isinstance(value, list) else []
        except (TypeError, ValueError):
            return []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_email": self.user.email if self.user else None,
            "query_text": self.query_text,
            "response_text": self.response_text,
            "retrieved_sections": self.sections(),
            "status": self.status,
            "error_message": self.error_message,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<QueryLog {self.id} user={self.user_id} {self.status}>"
