"""Build the eval catalog and gold queries from the product ingest log."""

import ast
import json
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_LOG_PATH = _ROOT / "logs" / "glancy.log"
_DIR = Path(__file__).resolve().parent

_PRODUCT_MARKER = "INFO - product:\n"

_SPECIFIC_LIMIT = 40
_PARTIAL_LIMIT = 10

_SPECIFIC_FIELDS = (
    ("item", "pr_item"),
    ("brand", "pr_brand"),
    ("colour", "pr_colour"),
    ("material", "pr_material"),
    ("range", "pr_range"),
)
_PARTIAL_FIELDS = (
    ("item", "pr_item"),
    ("brand", "pr_brand"),
)

_CATALOG_FIELDS = ("id", "name", "pr_item", "pr_brand", "pr_colour", "pr_material", "pr_range")


def _norm(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def parse_products(log_text: str) -> list[dict[str, str]]:
    """Return product dicts logged during ingest, in log order."""
    products = []
    start = 0
    while True:
        index = log_text.find(_PRODUCT_MARKER, start)
        if index < 0:
            break
        block_start = index + len(_PRODUCT_MARKER)
        lines = []
        consumed = 0
        for line in log_text[block_start:].splitlines(keepends=True):
            content = line.lstrip("\n")
            if not (content.startswith("{") or content.startswith("  ")):
                break
            lines.append(line)
            consumed += len(line)
        if not lines:
            raise ValueError("product marker was not followed by a dict")
        product = ast.literal_eval("".join(lines).strip())
        products.append({field: str(product[field]).strip() for field in _CATALOG_FIELDS})
        start = block_start + consumed
    if not products:
        raise ValueError(f"no products found in {_LOG_PATH}")
    # The ingest log writes the same page twice and upserts it once.
    unique: dict[str, dict[str, str]] = {}
    for product in products:
        unique[product["id"]] = product
    return list(unique.values())


def _group(products: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for product in products:
        groups[tuple(_norm(product[field]) for field in fields)].append(product)
    for group in groups.values():
        group.sort(key=lambda product: product["id"])
    return groups


def _representative(group: list[dict[str, str]]) -> dict[str, str]:
    return group[0]


def _search_text(product: dict[str, str], fields: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"{label}: {product[column].strip()}" for label, column in fields)


def _specific_message(product: dict[str, str]) -> str:
    return (
        f"{product['pr_colour'].strip()} "
        f"{_norm(product['pr_material'])} "
        f"{_norm(product['pr_item'])} "
        f"from {product['pr_brand'].strip()}, "
        f"{product['pr_range'].strip()}"
    )


def _partial_message(product: dict[str, str]) -> str:
    return f"{product['pr_item'].strip()} from {product['pr_brand'].strip()}"


def _round_robin(groups_by_brand: dict[str, list[list[dict[str, str]]]], limit: int) -> list[list[dict[str, str]]]:
    brands = sorted(groups_by_brand)
    pending = {brand: list(groups_by_brand[brand]) for brand in brands}
    chosen: list[list[dict[str, str]]] = []
    while len(chosen) < limit and any(pending.values()):
        for brand in brands:
            if not pending[brand]:
                continue
            chosen.append(pending[brand].pop(0))
            if len(chosen) == limit:
                break
    return chosen


def _specific_groups(products: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """One clean-colour group per item and brand, spread across brands."""
    five_fields = ("pr_item", "pr_brand", "pr_colour", "pr_material", "pr_range")
    by_item_brand: dict[tuple[str, str], list[dict[str, str]]] = {}
    for group in _group(products, five_fields).values():
        product = _representative(group)
        if "," in product["pr_colour"]:
            continue
        key = (_norm(product["pr_item"]), _norm(product["pr_brand"]))
        current = by_item_brand.get(key)
        if current is None or product["id"] < _representative(current)["id"]:
            by_item_brand[key] = group

    by_brand: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in by_item_brand.values():
        product = _representative(group)
        by_brand[_norm(product["pr_brand"])].append(group)
    for groups in by_brand.values():
        groups.sort(key=lambda group: (
            _norm(_representative(group)["pr_item"]),
            _norm(_representative(group)["pr_colour"]),
            _norm(_representative(group)["pr_range"]),
        ))
    return _round_robin(by_brand, _SPECIFIC_LIMIT)


def _partial_groups(products: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Item and brand pairs that match at least two products."""
    by_brand: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in _group(products, ("pr_item", "pr_brand")).values():
        if len(group) < 2:
            continue
        by_brand[_norm(_representative(group)["pr_brand"])].append(group)
    for groups in by_brand.values():
        groups.sort(key=lambda group: _norm(_representative(group)["pr_item"]))
    return _round_robin(by_brand, _PARTIAL_LIMIT)


def _query(number: int, kind: str, group: list[dict[str, str]]) -> dict:
    product = _representative(group)
    if kind == "specific":
        message = _specific_message(product)
        search_text = _search_text(product, _SPECIFIC_FIELDS)
    elif kind == "partial":
        message = _partial_message(product)
        search_text = _search_text(product, _PARTIAL_FIELDS)
    else:
        raise ValueError(f"unknown query kind: {kind}")
    relevant_ids = [item["id"] for item in group]
    if not relevant_ids:
        raise ValueError("relevant ids must not be empty")
    return {
        "id": f"q{number:03d}",
        "kind": kind,
        "message": message,
        "search_text": search_text,
        "relevant_ids": relevant_ids,
    }


def build_queries(products: list[dict[str, str]]) -> list[dict]:
    queries = []
    number = 1
    for group in _specific_groups(products):
        queries.append(_query(number, "specific", group))
        number += 1
    for group in _partial_groups(products):
        queries.append(_query(number, "partial", group))
        number += 1
    if not queries:
        raise ValueError("no eval queries could be built")
    return queries


def write_dataset(products: list[dict[str, str]], queries: list[dict]) -> None:
    (_DIR / "catalog.json").write_text(
        json.dumps(products, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (_DIR / "queries.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    products = parse_products(_LOG_PATH.read_text(encoding="utf-8"))
    queries = build_queries(products)
    write_dataset(products, queries)
    specific = sum(1 for query in queries if query["kind"] == "specific")
    partial = sum(1 for query in queries if query["kind"] == "partial")
    print(f"catalog: {len(products)} products")
    print(f"queries: {len(queries)} ({specific} specific, {partial} partial)")


if __name__ == "__main__":
    main()
