from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.database.mysql import mysql_connection
from logger import logger

router = APIRouter()

BATCH_SIZE = 500

_PRODUCT_COLUMNS = (
    "id",
    "name",
    "pr_item",
    "pr_brand",
    "pr_colour",
    "pr_material",
    "pr_range",
)

_REQUIRED_TEXT_COLUMNS = _PRODUCT_COLUMNS[1:]

_NON_EMPTY_FILTER = " AND ".join(
    "NULLIF(TRIM(" + column + "), '') IS NOT NULL" for column in _REQUIRED_TEXT_COLUMNS
)

_PRODUCT_QUERY = (
    "SELECT " + ", ".join(_PRODUCT_COLUMNS) + " "
    "FROM aos_products "
    "WHERE deleted = 0 "
    "AND id > %s "
    "AND " + _NON_EMPTY_FILTER + " "
    "ORDER BY id "
    "LIMIT %s"
)


class Product(BaseModel):
    id: str
    name: str | None = None
    pr_item: str | None = None
    pr_brand: str | None = None
    pr_colour: str | None = None
    pr_material: str | None = None
    pr_range: str | None = None


class ProductBatch(BaseModel):
    count: int
    next_after_id: str | None
    products: list[Product]


def fetch_product_batch(after_id: str = "", batch_size: int = BATCH_SIZE) -> list[dict[str, Any]]:
    """Return the next page of active products, ordered by id."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    with mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(_PRODUCT_QUERY, (after_id, batch_size))
            logger.info(f"fetched {_PRODUCT_QUERY} with {after_id} and {batch_size}")
            return list(cursor.fetchall())


def iter_product_batches(batch_size: int = BATCH_SIZE) -> Iterator[list[dict[str, Any]]]:
    """Yield active products from aos_products in keyset pages."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    after_id = ""
    with mysql_connection() as conn:
        with conn.cursor() as cursor:
            while True:
                cursor.execute(_PRODUCT_QUERY, (after_id, batch_size))
                logger.info(f"fetched {_PRODUCT_QUERY} with {after_id} and {batch_size}")
                batch = list(cursor.fetchall())
                if not batch:
                    return
                yield batch
                if len(batch) < batch_size:
                    return
                after_id = batch[-1]["id"]


@router.get("/products", response_model=ProductBatch)
def list_products(
    after_id: str = "",
    limit: int = Query(default=BATCH_SIZE, ge=1, le=BATCH_SIZE),
) -> ProductBatch:
    products = [Product.model_validate(row) for row in fetch_product_batch(after_id, limit)]
    next_after_id = products[-1].id if len(products) == limit else None
    return ProductBatch(count=len(products), next_after_id=next_after_id, products=products)


def main() -> None:
    total = 0
    for batch_number, batch in enumerate(iter_product_batches(), start=1):
        total += len(batch)
        print(f"batch {batch_number}: {len(batch)} products (total {total})")
    print(f"finished: {total} products")


if __name__ == "__main__":
    main()
