"""Provider-agnostic generation layer.

`PROVIDERS` is an ordered list of factories; the first success wins, later
entries act as fallbacks. A fine-tuned model can be prepended later without
touching the Flask app or the retrieval layer.

Transports (``GENERATION_BACKEND``):

* ``sdk``  — LangChain + langchain-google-genai (gRPC). Default locally.
* ``rest`` — plain HTTPS via requests, template rendered with str.format.
             Avoids importing langchain/gRPC entirely (~150 MB lighter);
             recommended on memory-constrained hosts such as Render free.
"""
import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
REST_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
BACKEND = os.getenv("GENERATION_BACKEND", "sdk").strip().lower()


@dataclass
class GenerationResult:
    """Mimics a LangChain message: exposes `.content`."""

    content: str


class RestGemini:
    """Minimal Gemini client over plain HTTPS (no gRPC, no langchain)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def invoke(self, prompt: str) -> GenerationResult:
        response = requests.post(
            f"{REST_ENDPOINT}/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        candidates = response.json().get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Empty Gemini response: {response.text[:200]}")

        parts = candidates[0].get("content", {}).get("parts") or []
        return GenerationResult("".join(part.get("text", "") for part in parts))


def _rest_provider():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return RestGemini(api_key=key)


def _sdk_provider():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        request_timeout=30,
        max_retries=2,
    )


PROVIDERS = [_rest_provider] if BACKEND == "rest" else [_sdk_provider]


def _render(template: str, variables: dict) -> str:
    try:
        return template.format(**variables)
    except (KeyError, IndexError):
        # Template contains literal braces; fall back to sequential replace.
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered


def generate(template: str, **vars: Any):
    """Render a prompt and call the first provider that responds."""
    last_error: Exception | None = None

    for make in PROVIDERS:
        try:
            model = make()
            if BACKEND == "rest":
                return model.invoke(_render(template, vars))

            from langchain_core.prompts import ChatPromptTemplate

            return (ChatPromptTemplate.from_template(template) | model).invoke(vars)
        except Exception as exc:  # try the next provider
            last_error = exc
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error
