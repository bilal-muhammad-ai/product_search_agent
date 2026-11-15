"""Score hybrid, dense, and keyword retrieval against the gold queries."""

import logging
from collections.abc import Callable

from app.database.retriever import retrieve, search_by_keyword, search_by_similarity
from app.evals.average_precision.metric import mean_average_precision
from app.evals.dataset.load import Query, load_queries
from app.evals.hit.metric import mean_hit_at_k
from app.evals.mrr.metric import mean_reciprocal_rank
from app.evals.ndcg.metric import mean_ndcg_at_k
from app.evals.precision.metric import mean_precision_at_k
from app.evals.recall.metric import mean_recall_at_k

_K = 5
_CANDIDATE_K = 20

Ranking = tuple[list[str], set[str]]
Search = Callable[[Query], list[str]]


def _hybrid(query: Query) -> list[str]:
    products = retrieve(query.search_text, limit=_CANDIDATE_K, rerank_text=query.message)
    return [product.id for product in products]


def _dense(query: Query) -> list[str]:
    products = search_by_similarity(query.search_text, limit=_CANDIDATE_K)
    return [product.id for product in products]


def _keyword(query: Query) -> list[str]:
    products = search_by_keyword(query.search_text, limit=_CANDIDATE_K)
    return [product.id for product in products]


def _score(cases: list[Ranking]) -> dict[str, float]:
    return {
        "precision@5": mean_precision_at_k(cases, _K),
        "recall@5": mean_recall_at_k(cases, _K),
        "hit@5": mean_hit_at_k(cases, _K),
        "mrr": mean_reciprocal_rank(cases),
        "ndcg@5": mean_ndcg_at_k(cases, _K),
        "average_precision": mean_average_precision(cases),
        "recall@20": mean_recall_at_k(cases, _CANDIDATE_K),
        "hit@20": mean_hit_at_k(cases, _CANDIDATE_K),
    }


def _print_channel(name: str, query_count: int, scores: dict[str, float]) -> None:
    print(f"{name} ({query_count} queries)")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")


def main() -> None:
    logging.getLogger("glancy_similarity_search").setLevel(logging.WARNING)
    queries = load_queries()
    if not queries:
        raise ValueError("no eval queries loaded")
    channels: dict[str, Search] = {
        "hybrid": _hybrid,
        "dense": _dense,
        "keyword": _keyword,
    }
    for name, search in channels.items():
        cases = [(search(query), set(query.relevant_ids)) for query in queries]
        _print_channel(name, len(queries), _score(cases))


if __name__ == "__main__":
    main()
