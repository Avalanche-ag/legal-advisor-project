"""End-to-end verification of RBAC, logging, and admin APIs.

The heavy RAG service is stubbed so this runs without chromadb/torch or an
API key. Run:  python scripts/verify_rbac.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# --- stub the heavy RAG service before any route imports it ---
fake_rag = types.ModuleType("app.services.rag")
CALLS = {"count": 0}


def _fake_answer(query, history=""):
    CALLS["count"] += 1
    if "boom" in query.lower():
        raise RuntimeError("simulated model failure")
    return (
        "Under Section 379 of the Indian Penal Code, theft is punishable with "
        "imprisonment up to three years, or with fine, or both.",
        [
            {
                "section": 379,
                "section_title": "Punishment for theft",
                "chapter": 16,
                "chapter_title": "of theft",
                "content": "Whoever commits theft shall be punished...",
            }
        ],
    )


def _fake_report(queries, answers):
    path = Path(tempfile.gettempdir()) / "verify_report.pdf"
    path.write_bytes(b"%PDF-1.4\n% verification stub\n")
    return str(path)


fake_rag.answer_ipc_query = _fake_answer
fake_rag.generate_report = _fake_report
sys.modules["app.services.rag"] = fake_rag

from app import create_app, db  # noqa: E402
from app.models import QueryLog, User  # noqa: E402

PASS, FAIL = [], []


def check(label, condition, detail=""):
    (PASS if condition else FAIL).append(label)
    print(f"{'PASS' if condition else 'FAIL'}  {label}{(' — ' + str(detail)) if detail else ''}")


def main():
    tmp = tempfile.mkdtemp()
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "verify-secret",
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp}/verify.db",
        }
    )
    with app.app_context():
        db.create_all()
        admin = User(email="admin@test.dev", username="admin", role="admin")
        admin.set_password("adminpass123")
        db.session.add(admin)
        db.session.commit()

    anon = app.test_client()
    check("GET / renders", anon.get("/").status_code == 200)
    check("GET /login renders", anon.get("/login").status_code == 200)
    check("GET /register renders", anon.get("/register").status_code == 200)
    check("POST /ask without auth -> 401", anon.post("/ask", json={"query": "theft"}).status_code == 401)
    check(
        "GET /api/me without auth -> 401",
        anon.get("/api/me", headers={"Accept": "application/json"}).status_code == 401,
    )

    # --- registration ---
    reg = anon.post(
        "/register",
        json={"email": "user@test.dev", "username": "tester", "password": "userpass123"},
    )
    check("POST /register -> 201", reg.status_code == 201, reg.get_json())

    dup = anon.post(
        "/register",
        json={"email": "user@test.dev", "username": "tester2", "password": "userpass123"},
    )
    check("duplicate email -> 409", dup.status_code == 409)

    bad = anon.post(
        "/register",
        json={"email": "nope", "username": "ab", "password": "short"},
    )
    check("invalid registration -> 400", bad.status_code == 400)

    # --- login as standard user ---
    user_client = app.test_client()
    login = user_client.post("/login", json={"email": "user@test.dev", "password": "userpass123"})
    check("POST /login -> 200", login.status_code == 200, login.get_json())
    check("login sets role=user", login.get_json().get("user", {}).get("role") == "user")

    me = user_client.get("/api/me").get_json()
    check("/api/me returns profile", me.get("user", {}).get("email") == "user@test.dev", me)

    # --- /ask success + logging ---
    ask = user_client.post("/ask", json={"query": "punishment for theft"})
    payload = ask.get_json()
    check("POST /ask (auth) -> 200", ask.status_code == 200, payload)
    check("response contains answer", "Section 379" in (payload.get("answer") or ""))
    check("response contains sections", len(payload.get("sections") or []) == 1)
    check("response contains latency_ms", isinstance(payload.get("latency_ms"), (int, float)))

    with app.app_context():
        success_log = QueryLog.query.filter_by(status="SUCCESS").first()
        check("QueryLog SUCCESS persisted", success_log is not None)
        if success_log:
            check("log stores response text", "Section 379" in (success_log.response_text or ""))
            check("log stores citations JSON", len(success_log.sections()) == 1)
            check("log stores latency", (success_log.latency_ms or 0) > 0)

    # --- /ask failure + logging ---
    boom = user_client.post("/ask", json={"query": "boom now"})
    check("POST /ask (failure) -> 500", boom.status_code == 500)
    with app.app_context():
        failed = QueryLog.query.filter_by(status="FAILED").first()
        check("QueryLog FAILED persisted", failed is not None)
        if failed:
            check("failure stores error message", "simulated model failure" in (failed.error_message or ""))
            check("failure has no response text", not failed.response_text)

    # --- history ---
    history = user_client.get("/api/history").get_json()
    check("GET /api/history returns items", len(history.get("items", [])) == 2, history.get("total"))
    check("history is user-scoped", all(i["user_email"] == "user@test.dev" for i in history.get("items", [])))

    # --- report ---
    report = user_client.get("/report")
    check("GET /report -> 200 PDF", report.status_code == 200 and report.mimetype == "application/pdf")

    # --- admin protection ---
    check(
        "standard user blocked from admin API -> 403",
        user_client.get("/api/admin/users").status_code == 403,
    )
    check(
        "standard user blocked from dashboard -> 403",
        user_client.get("/admin/dashboard").status_code == 403,
    )

    # --- admin flows ---
    admin_client = app.test_client()
    admin_login = admin_client.post("/login", json={"email": "admin@test.dev", "password": "adminpass123"})
    check("admin login -> 200", admin_login.status_code == 200)

    dash = admin_client.get("/admin/dashboard")
    check("GET /admin/dashboard -> 200", dash.status_code == 200)

    users = admin_client.get("/api/admin/users").get_json().get("users", [])
    check("GET /api/admin/users lists users", len(users) == 2, users)
    check("users include query counts", any(u["query_count"] == 2 for u in users), users)

    stats = admin_client.get("/api/admin/stats").get_json()
    check("GET /api/admin/stats totals", stats.get("total_users") == 2 and stats.get("total_queries") == 2, stats)
    check("stats success rate 50%", stats.get("success_rate") == 50.0, stats)
    check("stats avg latency present", stats.get("avg_latency_ms") is not None, stats)

    logs = admin_client.get("/api/admin/logs").get_json().get("logs", [])
    check("GET /api/admin/logs returns all logs", len(logs) == 2, len(logs))
    filtered = admin_client.get("/api/admin/logs?status=FAILED").get_json().get("logs", [])
    check("logs filter by status works", len(filtered) == 1 and filtered[0]["status"] == "FAILED")
    by_latency = admin_client.get("/api/admin/logs?sort=latency").get_json().get("logs", [])
    check("logs sort by latency works", len(by_latency) == 2)

    user_id = next(u["id"] for u in users if u["email"] == "user@test.dev")
    promote = admin_client.patch(f"/api/admin/users/{user_id}/role", json={"role": "admin"})
    check("PATCH role -> 200", promote.status_code == 200 and promote.get_json()["user"]["role"] == "admin")

    bad_role = admin_client.patch(f"/api/admin/users/{user_id}/role", json={"role": "superuser"})
    check("PATCH invalid role -> 400", bad_role.status_code == 400)

    toggle = admin_client.patch(f"/api/admin/users/{user_id}/toggle-status")
    check("PATCH toggle-status -> deactivated", toggle.get_json()["user"]["is_active"] is False)

    # deactivated user cannot log in
    blocked_client = app.test_client()
    blocked = blocked_client.post("/login", json={"email": "user@test.dev", "password": "userpass123"})
    check("deactivated user blocked -> 403", blocked.status_code == 403)

    # self-deactivation guard
    admin_id = next(u["id"] for u in users if u["email"] == "admin@test.dev")
    self_toggle = admin_client.patch(f"/api/admin/users/{admin_id}/toggle-status")
    check("cannot deactivate own account -> 400", self_toggle.status_code == 400)

    # logout
    check("GET /logout redirects", admin_client.get("/logout").status_code == 302)

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED CHECKS:")
        for name in FAIL:
            print("  -", name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
