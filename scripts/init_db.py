"""Create database tables and seed the initial admin account.

Usage:
    python scripts/init_db.py

Environment variables (development defaults in brackets):
    ADMIN_EMAIL     [admin@local.dev]
    ADMIN_USERNAME  [admin]
    ADMIN_PASSWORD  [admin12345]
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app, db  # noqa: E402
from app.models import User  # noqa: E402


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()

        email = os.getenv("ADMIN_EMAIL", "admin@local.dev").strip().lower()
        username = os.getenv("ADMIN_USERNAME", "admin").strip()
        password = os.getenv("ADMIN_PASSWORD", "admin12345")

        existing = User.query.filter_by(email=email).first()
        if existing:
            if existing.role != "admin":
                existing.role = "admin"
                db.session.commit()
                print(f"Promoted existing user to admin: {email}")
            else:
                print(f"Admin already present: {email}")
            return

        admin = User(email=email, username=username, role="admin", is_active=True)
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        print(f"Created admin user: {email}")


if __name__ == "__main__":
    main()
