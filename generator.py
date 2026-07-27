"""Generation layer.

Calls a chat model to turn a prompt template + variables into a response.
The model is loaded from an API key in the environment (Gemini by default).
If that call fails, the function falls back to the next configured provider
so the app stays up when one backend is rate-limited or down.
"""
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def _gemini() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        request_timeout=30,   # fail fast instead of hanging >60s on Render
        max_retries=2,        # don't retry 5× with exponential backoff
    )


# First entry is the primary model, the rest are fallbacks.
# Add a fine-tuned endpoint here later and Gemini becomes the backup.
PROVIDERS = [_gemini]


def generate(template: str, **vars: Any):
    """Render a prompt and call the first model that responds."""
    prompt = ChatPromptTemplate.from_template(template)
    last_error = None

    for make in PROVIDERS:
        try:
            model = make()
            return (prompt | model).invoke(vars)
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}") from last_error