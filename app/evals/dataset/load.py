import json
from dataclasses import dataclass
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    pr_item: str
    pr_brand: str
    pr_colour: str
    pr_material: str
    pr_range: str


@dataclass(frozen=True)
class Query:
    id: str
    kind: str
    message: str
    search_text: str
    relevant_ids: tuple[str, ...]


def _read(name: str) -> list[dict]:
    return json.loads((_DIR / name).read_text(encoding="utf-8"))


def load_catalog() -> list[Product]:
    return [Product(**row) for row in _read("catalog.json")]


def load_queries() -> list[Query]:
    queries = []
    for row in _read("queries.json"):
        queries.append(
            Query(
                id=row["id"],
                kind=row["kind"],
                message=row["message"],
                search_text=row["search_text"],
                relevant_ids=tuple(row["relevant_ids"]),
            )
        )
    return queries
