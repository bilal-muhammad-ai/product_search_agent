import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client.models import Fusion, FusionQuery, Prefetch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.database.qdrant import get_client
from app.database.storeembeddings import (
    COLLECTION_NAME,
    DENSE_MODEL_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    embed_sparse_texts,
    require_hybrid_collection,
)
from app.models.search import RetrievedProduct
from logger import logger

load_dotenv()

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_LIMIT = 20

_FIELD_LABELS = (
    ("item", "pr_item"),
    ("brand", "pr_brand"),
    ("colour", "pr_colour"),
    ("material", "pr_material"),
    ("range", "pr_range"),
)

_dense_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(DENSE_MODEL_NAME, token=_require_env("HF_TOKEN"))
    return _dense_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME, token=_require_env("HF_TOKEN"))
    return _reranker


def _embed_dense(text: str) -> list[float]:
    vector = _get_dense_model().encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector.tolist()


def _payload_text(payload: dict[str, Any]) -> str:
    """Rebuild the labeled document stored for each product embedding."""
    return "\n".join(
        f"{label}: {str(payload.get(column) or '').strip()}"
        for label, column in _FIELD_LABELS
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _rerank_scores(query_text: str, documents: list[str]) -> list[float]:
    scores = _get_reranker().predict(
        [(query_text, document) for document in documents],
        show_progress_bar=False,
    )
    return [float(score) for score in scores]


def retrieve(
    query_text: str,
    limit: int = 5,
    rerank_text: str | None = None,
) -> list[RetrievedProduct]:
    """Embed a query, fuse dense and sparse hits, and return the reranked products.

    Dense and sparse search use query_text. The cross-encoder uses rerank_text
    when it is provided, otherwise it uses query_text.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    text = query_text.strip()
    if not text:
        raise ValueError("query_text must not be empty")
    rank_text = (rerank_text if rerank_text is not None else query_text).strip()
    if not rank_text:
        raise ValueError("rerank_text must not be empty")
    logger.info("retrieving products for query: %s", text)
    if rank_text != text:
        logger.info("reranking products for query: %s", rank_text)

    require_hybrid_collection()
    dense = _embed_dense(text)
    sparse = embed_sparse_texts([text])[0]
    candidate_limit = max(CANDIDATE_LIMIT, limit)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=candidate_limit),
            Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=candidate_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=candidate_limit,
        with_payload=True,
        with_vectors=False,
    )
    logger.info("response: %s", response)
    points = response.points
    if not points:
        return []

    documents = [_payload_text(point.payload or {}) for point in points]
    ranked = sorted(
        zip(points, _rerank_scores(rank_text, documents), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    products = []
    for point, score in ranked[:limit]:
        payload = point.payload or {}
        products.append(
            RetrievedProduct(
                id=str(point.id),
                score=score,
                pr_item=_optional_text(payload.get("pr_item")),
                pr_brand=_optional_text(payload.get("pr_brand")),
                pr_colour=_optional_text(payload.get("pr_colour")),
                pr_material=_optional_text(payload.get("pr_material")),
                pr_range=_optional_text(payload.get("pr_range")),
            )
        )
    logger.info("retrieved %s products", len(products))
    return products


def _format_points(points: list[Any]) -> str:
    """One readable line per hit: score, brand, item, colour, material, range."""
    lines = []
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        fields = " | ".join(
            str(payload.get(column) or "-")
            for column in ("pr_brand", "pr_item", "pr_colour", "pr_material", "pr_range")
        )
        lines.append(f"{index}. {float(point.score):.4f}  {fields}")
    return "\n".join(lines) or "(no hits)"


def _products_from_points(points: list[Any]) -> list[RetrievedProduct]:
    """Map Qdrant hits to products, keeping each point's own score."""
    products = []
    for point in points:
        payload = point.payload or {}
        products.append(
            RetrievedProduct(
                id=str(point.id),
                score=float(point.score),
                pr_item=_optional_text(payload.get("pr_item")),
                pr_brand=_optional_text(payload.get("pr_brand")),
                pr_colour=_optional_text(payload.get("pr_colour")),
                pr_material=_optional_text(payload.get("pr_material")),
                pr_range=_optional_text(payload.get("pr_range")),
            )
        )
    return products


def _debug_query_text(query_text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    text = query_text.strip()
    if not text:
        raise ValueError("query_text must not be empty")
    require_hybrid_collection()
    return text


def search_by_similarity(query_text: str, limit: int = 5) -> list[RetrievedProduct]:
    """Return dense cosine hits for debugging. Does not fuse or rerank."""
    text = _debug_query_text(query_text, limit)
    logger.info("similarity search for query: %s", query_text)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=_embed_dense(text),
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    logger.info("similarity response:\n%s", _format_points(response.points))
    products = _products_from_points(response.points)
    logger.info("similarity search returned %s products", len(products))
    return products


def search_by_keyword(query_text: str, limit: int = 5) -> list[RetrievedProduct]:
    """Return sparse BM25 hits for debugging. Does not fuse or rerank."""
    text = _debug_query_text(query_text, limit)
    logger.info("keyword search for query: %s", query_text)
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_sparse_texts([text])[0],
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    logger.info("keyword response:\n%s", _format_points(response.points))
    products = _products_from_points(response.points)
    logger.info("keyword search returned %s products", len(products))
    return products


