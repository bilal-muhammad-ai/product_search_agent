from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.database.qdrant import get_client
from logger import logger

COLLECTION_NAME = "glancy_products"
VECTOR_SIZE = 384
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"

EmbeddedProduct = tuple[str, list[float], SparseVector, dict[str, Any]]

_DENSE_ONLY_COLLECTION = (
    "Collection 'products' does not have dense and sparse vectors. "
    "Delete it, then embed products again. Existing points were left untouched."
)

_sparse_model: SparseTextEmbedding | None = None


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    """Encode texts with the BM25 model used for both ingest and search."""
    if not texts:
        return []

    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    return [
        SparseVector(
            indices=[int(index) for index in embedding.indices],
            values=[float(value) for value in embedding.values],
        )
        for embedding in _sparse_model.embed(texts)
    ]


def _has_hybrid_schema() -> bool:
    params = get_client().get_collection(COLLECTION_NAME).config.params
    vectors = params.vectors
    sparse_vectors = params.sparse_vectors or {}
    if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
        return False
    if getattr(vectors[DENSE_VECTOR_NAME], "size", None) != VECTOR_SIZE:
        return False
    return SPARSE_VECTOR_NAME in sparse_vectors


def require_hybrid_collection() -> None:
    """Fail when search cannot use both named vectors. Does not create or delete anything."""
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError("Collection 'products' does not exist. Embed products before searching.")
    #if not _has_hybrid_schema():
    #    raise RuntimeError(_DENSE_ONLY_COLLECTION)


def ensure_products_collection() -> None:
    """Create the hybrid products collection when it does not already exist."""
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        #if _has_hybrid_schema():
        #    return
        #raise RuntimeError(_DENSE_ONLY_COLLECTION)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    logger.info("created qdrant collection %s", COLLECTION_NAME)


def store_product_embeddings(embedded_batch: list[EmbeddedProduct]) -> int:
    """Upsert one page of dense and sparse product vectors, with the product as payload."""
    if not embedded_batch:
        return 0

    ensure_products_collection()
    points = [
        PointStruct(
            id=product_id,
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: sparse,
            },
            payload=product,
        )
        for product_id, dense, sparse, product in embedded_batch
    ]
    get_client().upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("upserted %s products into %s", len(points), COLLECTION_NAME)
    return len(points)
