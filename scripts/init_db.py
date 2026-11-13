"""Create database tables and seed the initial admin account.

Safe to run repeatedly (idempotent). The app also performs this bootstrap at
startup, so this script is mainly for local/CLI use.

Usage:
    python scripts/init_db.py

Environment variables (development defaults in brackets):
    ADMIN_EMAIL     [admin@local.dev]
    ADMIN_USERNAME  [admin]
    ADMIN_PASSWORD  [admin12345]

If ADMIN_PASSWORD is set explicitly, an existing admin's password is updated to
match — useful when the account was seeded earlier with a different value.
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

        email = (os.getenv("ADMIN_EMAIL") or "admin@local.dev").strip().lower()
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD")

        existing = User.query.filter_by(email=email).first()
        if existing:
            if existing.role != "admin":
                existing.role = "admin"
                print(f"Promoted existing user to admin: {email}")

            if password:
                existing.set_password(password)
                print(f"Reset admin password from ADMIN_PASSWORD: {email}")
            else:
                print(
                    f"Admin already present: {email} "
                    "(set ADMIN_PASSWORD to also reset the password)"
                )
            db.session.commit()
            return

        admin = User(
            email=email,
            username=username,
            role="admin",
            is_active=True,
        )
        admin.set_password(password or "admin12345")
        db.session.add(admin)
        db.session.commit()
        print(f"Created admin user: {email}")

        if not password:
            print("WARNING: using the default password 'admin12345'. Set "
                  "ADMIN_PASSWORD before exposing this app publicly.")


if __name__ == "__main__":
    main()
