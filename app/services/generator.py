"""Provider-agnostic generation layer.

`PROVIDERS` is an ordered list of factories. The first provider whose call
succeeds wins; later entries act as fallbacks when one backend is
rate-limited or unavailable. A fine-tuned model can be prepended later
without touching the Flask app or the retrieval layer.
"""
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _gemini() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        request_timeout=30,
        max_retries=2,
    )


PROVIDERS = [_gemini]


def generate(template: str, **vars: Any):
    """Render a prompt and call the first model that responds."""
    prompt = ChatPromptTemplate.from_template(template)
    last_error: Exception | None = None

    for make in PROVIDERS:
        try:
            return (prompt | make()).invoke(vars)
        except Exception as exc:  # try the next provider
            last_error = exc
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error
