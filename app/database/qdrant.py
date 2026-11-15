import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_client() -> QdrantClient:
    """Open a client to the local Docker Qdrant service (QDRANT_URL)."""
    return QdrantClient(
        url=_require_env("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
    )
