"""Multi-provider LLM generation with a 3-tier fallback queue.

Order (first success wins):

    1. Groq        — free tier, fastest              (GROQ_API_KEY)
    2. MiniMax     — very low cost, OpenAI-compatible (MINIMAX_API_KEY)
    3. Gemini      — free tier, high availability     (GEMINI_API_KEY)

Every call goes through ``requests``. No ``openai`` / ``groq`` /
``google-generativeai`` / ``langchain`` imports, which keeps peak RSS low
enough for Render's 512 MB free tier.

The retrieval layer supplies the legal knowledge: the prompt already contains
the top-3 IPC sections. Providers are used purely as language models — none of
them performs a web search or knows the IPC.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a single provider cannot produce an answer."""


@dataclass
class GenerationResult:
    """Mimics a LangChain chat message: exposes ``.content``."""

    content: str
    provider: str = ""


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------
def _post(url: str, *, headers: dict, payload: dict, timeout: tuple[int, int]) -> dict:
    """POST JSON and return the decoded body, translating failures to ProviderError."""
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise ProviderError(f"timeout after {timeout[1]}s") from exc
    except requests.RequestException as exc:
        raise ProviderError(f"network error: {exc}") from exc

    if response.status_code == 429:
        raise ProviderError("rate limited (429)")
    if response.status_code >= 400:
        raise ProviderError(f"HTTP {response.status_code}: {response.text[:160]}")

    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"invalid JSON: {response.text[:160]}") from exc


def _openai_content(data: dict) -> str:
    """Extract text from an OpenAI-compatible chat-completions payload."""
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("no choices in response")
    message = choices[0].get("message") or {}
    text = message.get("content") or choices[0].get("text") or ""
    if isinstance(text, list):  # some gateways return content parts
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return str(text).strip()


# --------------------------------------------------------------------------
# 1. Groq
# --------------------------------------------------------------------------
def _call_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ProviderError("GROQ_API_KEY not configured")

    data = _post(
        os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        },
        timeout=(5, int(os.getenv("GROQ_TIMEOUT", "12"))),
    )
    return _openai_content(data)


# --------------------------------------------------------------------------
# 2. MiniMax (OpenAI-compatible v2 endpoint)
# --------------------------------------------------------------------------
def _call_minimax(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ProviderError("MINIMAX_API_KEY not configured")

    data = _post(
        os.getenv(
            "MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
        ),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": os.getenv("MINIMAX_MODEL", "abab6.5s"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        },
        timeout=(5, int(os.getenv("MINIMAX_TIMEOUT", "15"))),
    )

    # MiniMax reports application-level errors inside a 200 response.
    status = (data.get("base_resp") or {}).get("status_code")
    if status not in (None, 0):
        raise ProviderError(
            f"status_code={status} {(data.get('base_resp') or {}).get('status_msg', '')}".strip()
        )

    try:
        return _openai_content(data)
    except ProviderError:
        reply = data.get("reply")
        if reply:
            return str(reply).strip()
        raise


# --------------------------------------------------------------------------
# 3. Gemini
# --------------------------------------------------------------------------
def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY not configured")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        # API key goes in a header so it never appears in URLs or access logs.
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        payload={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
                "maxOutputTokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
            },
        },
        timeout=(5, int(os.getenv("GEMINI_TIMEOUT", "15"))),
    )

    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise ProviderError(f"no candidates (blockReason={feedback.get('blockReason')})")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise ProviderError(f"empty completion (finishReason={candidates[0].get('finishReason')})")
    return text


# Ordered fallback queue. Every entry is (name, callable(system, user) -> str).
PROVIDERS: list[tuple[str, Callable[[str, str], str]]] = [
    ("groq", _call_groq),
    ("minimax", _call_minimax),
    ("gemini", _call_gemini),
]


def generate_answer(system_prompt: str, user_prompt: str) -> GenerationResult:
    """Try each provider in order; return the first successful answer."""
    errors: list[str] = []

    for name, provider_fn in PROVIDERS:
        started = time.perf_counter()
        try:
            logger.info("Attempting generation with provider: %s", name)
            text = provider_fn(system_prompt, user_prompt)
            if not text:
                raise ProviderError("empty response")

            elapsed_ms = round((time.perf_counter() - started) * 1000)
            logger.info("Provider %s succeeded in %s ms", name, elapsed_ms)
            return GenerationResult(content=text, provider=name)

        except Exception as exc:  # any failure moves us down the queue
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            logger.warning(
                "Provider %s failed after %s ms: %s — trying next", name, elapsed_ms, exc
            )
            errors.append(f"{name}: {exc}")
            continue

    raise RuntimeError("All LLM providers failed. Details: " + "; ".join(errors))


def generate(template: str, **vars) -> GenerationResult:
    """Backward-compatible wrapper used by the RAG pipeline.

    Renders ``template`` and splits it into a system instruction block and a
    user block so providers receive a proper chat-shaped request.
    """
    rendered = _render(template, vars)
    system_prompt, user_prompt = _split_prompt(rendered)
    return generate_answer(system_prompt, user_prompt)


def _render(template: str, variables: dict) -> str:
    try:
        return template.format(**variables)
    except (KeyError, IndexError):
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered


def _split_prompt(rendered: str) -> tuple[str, str]:
    """Use the instruction paragraph as the system message, the rest as user."""
    marker = "Previous Conversation:"
    if marker in rendered:
        head, tail = rendered.split(marker, 1)
        return head.strip(), (marker + tail).strip()
    return DEFAULT_SYSTEM_PROMPT, rendered.strip()


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI legal assistant specializing in Indian law and the Indian "
    "Penal Code (IPC). Answer only from the supplied IPC context, cite the "
    "section number for every claim, and say you don't know if the context is "
    "insufficient."
)
