"""nDCG@k with binary gain 1 / log2(rank + 1)."""

import math


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("relevant must not be empty")
    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / math.log2(index + 2)
        for index, doc_id in enumerate(retrieved[:k])
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(cases: list[tuple[list[str], set[str]]], k: int) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(ndcg_at_k(retrieved, relevant, k) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    score = ndcg_at_k(retrieved, relevant, 5)
    missed = ndcg_at_k(["b"], relevant, 5)
    mean = mean_ndcg_at_k([(retrieved, relevant), (["b"], relevant)], 5)
    assert math.isclose(score, expected)
    assert missed == 0.0
    assert math.isclose(mean, expected / 2)
    print(f"ndcg@5 example: {score}")
    print(f"mean ndcg@5 example: {mean}")
