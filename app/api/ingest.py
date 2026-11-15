import os
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.api.ingest_products import iter_product_batches
from app.database.storeembeddings import (
    DENSE_MODEL_NAME,
    EmbeddedProduct,
    embed_sparse_texts,
    store_product_embeddings,
)
from logger import logger, pretty
from fastapi import APIRouter

router = APIRouter()

load_dotenv()

MODEL_NAME = DENSE_MODEL_NAME

_FIELD_LABELS = (
    ("item", "pr_item"),
    ("brand", "pr_brand"),
    ("colour", "pr_colour"),
    ("material", "pr_material"),
    ("range", "pr_range"),
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def clean_multienum(value: Any) -> str:
    """Turn a SuiteCRM multienum like '^Aegean Blue^,^Aqua^' into 'Aegean Blue, Aqua'."""
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip().strip("^").strip() for part in text.split(",")]
    return ", ".join(part for part in parts if part)


def _prepare_product(row: dict[str, Any]) -> dict[str, Any]:
    """Copy a product row with colour cleaned for embedding text and filter metadata."""
    product = dict(row)
    product["pr_colour"] = clean_multienum(product.get("pr_colour"))
    return product


def product_embedding_text(product: dict[str, Any]) -> str:
    """Join the five attribute fields into one embedding document."""
    logger.info("product:\n%s", pretty(product))
    lines = [
        f"{label}: {str(product.get(column) or '').strip()}"
        for label, column in _FIELD_LABELS
    ]
    text = "\n".join(lines)
    logger.info("embedding text:\n%s", text)
    return text


def load_embedding_model() -> SentenceTransformer:
    """Load the local Hugging Face embedding model using HF_TOKEN."""
    return SentenceTransformer(MODEL_NAME, token=_require_env("HF_TOKEN"))


def _embed_batch(
    model: SentenceTransformer,
    batch: list[dict[str, Any]],
) -> list[EmbeddedProduct]:
    products = [_prepare_product(row) for row in batch]
    texts = [product_embedding_text(product) for product in products]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sparse_vectors = embed_sparse_texts(texts)
    return [
        (str(product["id"]), vector.tolist(), sparse, product)
        for product, vector, sparse in zip(products, vectors, sparse_vectors, strict=True)
    ]


def iter_embedded_batches() -> Iterator[list[EmbeddedProduct]]:
    """Embed each keyset page and yield that page of dense and sparse product rows."""
    model = load_embedding_model()
    # for batch in iter_product_batches():
    #     yield _embed_batch(model, batch)
    first_batch = next(iter_product_batches(), None)
    if first_batch is not None:
        yield _embed_batch(model, first_batch)


def embed_product_batches() -> Iterator[EmbeddedProduct]:
    """Embed every active product, yielding one dense and sparse vector at a time."""
    for embedded_batch in iter_embedded_batches():
        yield from embedded_batch


@router.post("/embed_products")
def embed_products() -> dict[str, Any]:
    """Embed the current test batch and store dense and sparse vectors in Qdrant."""
    total = 0
    dimension = 0
    stored = 0
    for batch_number, embedded_batch in enumerate(iter_embedded_batches(), start=1):
        total += len(embedded_batch)
        if embedded_batch:
            dimension = len(embedded_batch[0][1])
        stored += store_product_embeddings(embedded_batch)
        logger.info(
            f"batch {batch_number}: {len(embedded_batch)} products "
            f"(total {total}, dim {dimension})"
        )
    logger.info(f"finished: {total} products, dim {dimension}, stored {stored}")
    return {"total": total, "dimension": dimension, "stored": stored}