"""Precision@k: fraction of the top k retrieved ids that are relevant."""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    return len(set(retrieved[:k]) & relevant) / k


def mean_precision_at_k(cases: list[tuple[list[str], set[str]]], k: int) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(precision_at_k(retrieved, relevant, k) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    score = precision_at_k(retrieved, relevant, 5)
    mean = mean_precision_at_k([(retrieved, relevant), (["b"], relevant)], 5)
    assert score == 0.4
    assert mean == 0.2
    print(f"precision@5 example: {score}")
    print(f"mean precision@5 example: {mean}")
