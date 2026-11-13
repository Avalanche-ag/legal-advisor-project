"""Verify the 3-tier provider fallback queue without spending API credits.

All HTTP is faked by monkeypatching ``requests.post``; the real provider
functions, parsing and ordering logic are exercised.

Usage:
    python scripts/verify_fallback.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.services import generator  # noqa: E402

PASS, FAIL = [], []


def check(label, condition, detail=""):
    (PASS if condition else FAIL).append(label)
    print(f"{'PASS' if condition else 'FAIL'}  {label}{(' — ' + str(detail)) if detail else ''}")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(self._payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def fake_post_factory(behaviour):
    """behaviour: dict provider -> 'ok' | '429' | '500' | 'timeout'."""
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        if "groq.com" in url:
            name = "groq"
        elif "minimaxi" in url or "minimax" in url:
            name = "minimax"
        else:
            name = "gemini"

        calls.append(name)
        mode = behaviour.get(name, "ok")

        if mode == "timeout":
            import requests

            raise requests.Timeout("simulated timeout")
        if mode == "429":
            return FakeResponse(429, {"error": "rate limited"})
        if mode == "500":
            return FakeResponse(500, {"error": "boom"})

        if name == "gemini":
            return FakeResponse(
                200, {"candidates": [{"content": {"parts": [{"text": "gemini answer"}]}}]}
            )
        return FakeResponse(
            200, {"choices": [{"message": {"content": f"{name} answer"}}]}
        )

    return fake_post, calls


def run(behaviour, keys=("GROQ_API_KEY", "MINIMAX_API_KEY", "GEMINI_API_KEY")):
    import os

    for key in ("GROQ_API_KEY", "MINIMAX_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(key, None)
    for key in keys:
        os.environ[key] = "test-key"

    original = generator.requests.post
    fake_post, calls = fake_post_factory(behaviour)
    generator.requests.post = fake_post
    try:
        result = generator.generate_answer("system", "user")
        return result, calls, None
    except Exception as exc:
        return None, calls, exc
    finally:
        generator.requests.post = original


def main() -> int:
    # 1. Groq healthy -> only Groq is called
    result, calls, err = run({})
    check("groq success returns groq answer", result and result.provider == "groq", result)
    check("groq success calls only groq", calls == ["groq"], calls)

    # 2. Groq rate-limited -> MiniMax wins
    result, calls, err = run({"groq": "429"})
    check("groq 429 falls back to minimax", result and result.provider == "minimax", result)
    check("groq 429 call order", calls == ["groq", "minimax"], calls)

    # 3. Groq + MiniMax down -> Gemini wins
    result, calls, err = run({"groq": "500", "minimax": "timeout"})
    check("groq+minimax fail falls back to gemini", result and result.provider == "gemini", result)
    check("full fallback call order", calls == ["groq", "minimax", "gemini"], calls)

    # 4. Everything fails -> RuntimeError listing every provider
    result, calls, err = run({"groq": "429", "minimax": "500", "gemini": "500"})
    check("all providers fail raises RuntimeError", isinstance(err, RuntimeError), err)
    check(
        "error message names every provider",
        err and all(name in str(err) for name in ("groq", "minimax", "gemini")),
        err,
    )

    # 5. Missing key is skipped without an HTTP call
    result, calls, err = run({}, keys=("MINIMAX_API_KEY", "GEMINI_API_KEY"))
    check("missing groq key is skipped", result and result.provider == "minimax", result)
    check("missing key causes no groq HTTP call", calls == ["minimax"], calls)

    # 6. Only Gemini configured
    result, calls, err = run({}, keys=("GEMINI_API_KEY",))
    check("gemini-only config works", result and result.provider == "gemini", result)
    check("gemini-only call order", calls == ["gemini"], calls)

    # 7. MiniMax application-level error inside HTTP 200
    import os

    for key in ("GROQ_API_KEY", "MINIMAX_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(key, None)
    os.environ["MINIMAX_API_KEY"] = "k"
    os.environ["GEMINI_API_KEY"] = "k"
    original = generator.requests.post

    def minimax_app_error(url, headers=None, json=None, timeout=None):
        if "minimax" in url:
            return FakeResponse(200, {"base_resp": {"status_code": 1004, "status_msg": "bad auth"}})
        return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "gemini answer"}]}}]})

    generator.requests.post = minimax_app_error
    try:
        result = generator.generate_answer("s", "u")
        check("minimax base_resp error falls through", result.provider == "gemini", result)
    finally:
        generator.requests.post = original

    # 8. Legacy wrapper still splits system/user and returns .content
    _, calls, _ = run({})
    result = None
    original = generator.requests.post
    captured = {}

    def capture(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    generator.requests.post = capture
    try:
        import os as _os

        _os.environ["GROQ_API_KEY"] = "k"
        out = generator.generate(
            "SYSTEM RULES\n\nPrevious Conversation:\n{history}\n\nQuestion: {question}\n\nContext:\n{context}",
            history="",
            question="punishment for theft",
            context="Section 379: ...",
        )
        check("generate() returns GenerationResult", hasattr(out, "content"), type(out))
        messages = captured.get("messages") or []
        check("system role populated", messages and messages[0]["role"] == "system", messages[:1])
        check("user role populated", len(messages) == 2 and messages[1]["role"] == "user")
        check("question reaches user message", "punishment for theft" in messages[1]["content"])
    finally:
        generator.requests.post = original

    # 9. MiniMax endpoint resolution
    import os as _os

    _os.environ.pop("MINIMAX_BASE_URL", None)
    urls = generator._minimax_urls()
    check(
        "minimax default endpoints: openai first, legacy second",
        urls[0].endswith("/v1/chat/completions") and "chatcompletion_v2" in urls[1],
        urls,
    )

    _os.environ["MINIMAX_BASE_URL"] = "https://api.minimax.io/v1"
    check(
        "minimax bare base override appends /chat/completions",
        generator._minimax_urls() == ["https://api.minimax.io/v1/chat/completions"],
        generator._minimax_urls(),
    )

    _os.environ["MINIMAX_BASE_URL"] = "https://example.test/custom/endpoint"
    check(
        "minimax full URL override used verbatim",
        generator._minimax_urls() == ["https://example.test/custom/endpoint"],
        generator._minimax_urls(),
    )
    _os.environ.pop("MINIMAX_BASE_URL", None)

    # 10. 404 on the primary endpoint falls through to the legacy endpoint
    _os.environ["MINIMAX_API_KEY"] = "k"
    seen = []

    def fake_post(url, headers=None, payload=None, timeout=None):
        seen.append(url)
        if len(seen) == 1:
            raise generator.EndpointNotFound(f"HTTP 404 for {url}")
        return {"choices": [{"message": {"content": "legacy endpoint answer"}}]}

    original_post = generator._post
    generator._post = fake_post
    try:
        text = generator._call_minimax("system", "user")
        check("minimax 404 falls back to legacy endpoint", text == "legacy endpoint answer", text)
        check("minimax tried both endpoints in order", len(seen) == 2, seen)
    finally:
        generator._post = original_post

    # 11. Request payload conforms to the OpenAI chat-completions shape
    captured = {}

    def capture_post(url, headers=None, payload=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["json"] = payload or {}
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "ok"}}]}

    generator._post = capture_post
    try:
        generator._call_minimax("SYSTEM RULES", "user question")
    finally:
        generator._post = capture_post
        generator._post = original_post

    body = captured.get("json", {})
    check("minimax payload has model", "model" in body, body.keys())
    check(
        "minimax payload messages = [system, user]",
        [m["role"] for m in body.get("messages", [])] == ["system", "user"],
        body.get("messages"),
    )
    check("minimax payload temperature present", "temperature" in body)
    check(
        "minimax auth header is Bearer",
        str(captured.get("headers", {}).get("Authorization", "")).startswith("Bearer "),
    )
    check("minimax timeout tuple is (5, 15)", captured.get("timeout") == (5, 15), captured.get("timeout"))

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print("  -", name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
