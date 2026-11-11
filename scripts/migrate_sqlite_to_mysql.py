"""Copy users and query logs from the local SQLite database into MySQL.

Usage:
    # 1. Point at the target database
    export MYSQL_HOST=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DATABASE=ipc_app
    # 2. Run
    python scripts/migrate_sqlite_to_mysql.py [--sqlite instance/ipc_app.db] [--dry-run]

The script is idempotent:
  * users are matched by email and skipped if already present;
  * query logs are matched by (user email, query text, created_at) tuple.
"""
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app, db  # noqa: E402
from app.models import QueryLog, User  # noqa: E402


def parse_dt(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", default=str(BASE_DIR / "instance" / "ipc_app.db"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        print(f"Source database not found: {sqlite_path}")
        return 1

    app = create_app()
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
        print("Refusing to run: target DATABASE is SQLite. Set MYSQL_* env vars first.")
        return 1

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    users = [dict(row) for row in conn.execute("SELECT * FROM user")]
    logs = [dict(row) for row in conn.execute("SELECT * FROM query_log")]

    print(f"Source: {sqlite_path} ({len(users)} users, {len(logs)} logs)")
    print(f"Target: {app.config['SQLALCHEMY_DATABASE_URI'].split('@')[-1]}")

    if args.dry_run:
        print("Dry run — no writes performed.")
        return 0

    with app.app_context():
        db.create_all()

        id_map = {}
        created_users = 0
        for row in users:
            existing = User.query.filter_by(email=row["email"]).first()
            if existing:
                id_map[row["id"]] = existing.id
                continue
            user = User(
                email=row["email"],
                username=row["username"],
                password_hash=row["password_hash"],
                role=row.get("role") or "user",
                is_active=bool(row.get("is_active", 1)),
                created_at=parse_dt(row.get("created_at")) or datetime.utcnow(),
            )
            db.session.add(user)
            db.session.flush()
            id_map[row["id"]] = user.id
            created_users += 1
        db.session.commit()

        created_logs = 0
        skipped_logs = 0
        for row in logs:
            new_user_id = id_map.get(row["user_id"])
            if not new_user_id:
                skipped_logs += 1
                continue
            created_at = parse_dt(row.get("created_at")) or datetime.utcnow()
            duplicate = QueryLog.query.filter_by(
                user_id=new_user_id,
                query_text=row["query_text"],
                created_at=created_at,
            ).first()
            if duplicate:
                skipped_logs += 1
                continue
            db.session.add(
                QueryLog(
                    user_id=new_user_id,
                    query_text=row["query_text"],
                    response_text=row.get("response_text"),
                    retrieved_sections=row.get("retrieved_sections"),
                    status=row.get("status") or "SUCCESS",
                    error_message=row.get("error_message"),
                    latency_ms=row.get("latency_ms"),
                    created_at=created_at,
                )
            )
            created_logs += 1
        db.session.commit()

    print(f"Migrated users: {created_users} (mapped {len(id_map)})")
    print(f"Migrated logs : {created_logs} (skipped {skipped_logs})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
