"""Hit@k: 1 when any relevant id appears in the top k, otherwise 0."""


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    return float(any(doc_id in relevant for doc_id in retrieved[:k]))


def mean_hit_at_k(cases: list[tuple[list[str], set[str]]], k: int) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(hit_at_k(retrieved, relevant, k) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    score = hit_at_k(retrieved, relevant, 5)
    mean = mean_hit_at_k([(retrieved, relevant), (["b"], relevant)], 5)
    assert score == 1.0
    assert mean == 0.5
    print(f"hit@5 example: {score}")
    print(f"mean hit@5 example: {mean}")
