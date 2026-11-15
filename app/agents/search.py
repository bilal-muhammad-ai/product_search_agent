import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.models.search import RewrittenQuery
from logger import logger

load_dotenv()

_FIELD_LABELS = ("item", "brand", "colour", "material", "range")

_SYSTEM_PROMPT = """You are a product search query optimizer.

Extract the core product search query from the
user's message.

Rules:
1. Remove conversational filler.
2. Preserve product names and attributes.
3. Preserve brand, colour, material and range.
4. Do not invent missing product attributes.
5. Return structured JSON.


Fill a field only when the customer stated it or clearly implied it.
Do not invent a brand, material, range, or item.
Keep colour in the customer's own words.
Use an empty string for any field that is not present.
"""

_rewriter: Runnable | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_rewriter() -> Runnable:
    """Build the structured rewrite model once, failing if Groq is not configured."""
    global _rewriter
    if _rewriter is None:
        _require_env("GROQ_API_KEY")
        _rewriter = init_chat_model(
            _require_env("MODEL_NAME"),
            model_provider=_require_env("MODEL_PROVIDER"),
            temperature=0,
            max_tokens=int(os.getenv("MAX_TOKENS", "800")),
        ).with_structured_output(RewrittenQuery)
    return _rewriter


def _clean(value: str | None) -> str:
    return str(value or "").strip()


def format_labeled_text(fields: RewrittenQuery, fallback: str) -> str:
    """Join extracted attributes using the same labels as product embeddings."""
    lines = [
        f"{label}: {value}"
        for label in _FIELD_LABELS
        if (value := _clean(getattr(fields, label)))
    ]
    return "\n".join(lines) or fallback


def rewrite_query(message: str) -> RewrittenQuery:
    text = message.strip()

    if not text:
        return RewrittenQuery(intent="unknown")

    parsed = _get_rewriter().invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ])

    if parsed is None:
        raise RuntimeError("Query rewrite model returned no result")

    return (
        parsed
        if isinstance(parsed, RewrittenQuery)
        else RewrittenQuery.model_validate(parsed)
    )
