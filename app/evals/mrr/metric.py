"""Reciprocal rank of the first relevant id. Zero when none are retrieved."""


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(cases: list[tuple[list[str], set[str]]]) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    score = reciprocal_rank(retrieved, relevant)
    mean = mean_reciprocal_rank([(retrieved, relevant), (["b", "a"], relevant)])
    assert score == 1.0
    assert mean == 0.75
    print(f"reciprocal rank example: {score}")
    print(f"mean reciprocal rank example: {mean}")
