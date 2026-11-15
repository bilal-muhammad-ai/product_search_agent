"""Recall@k: fraction of the relevant ids that appear in the top k."""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("relevant must not be empty")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mean_recall_at_k(cases: list[tuple[list[str], set[str]]], k: int) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    score = recall_at_k(retrieved, relevant, 5)
    mean = mean_recall_at_k([(retrieved, relevant), (["b"], relevant)], 5)
    assert score == 1.0
    assert mean == 0.5
    print(f"recall@5 example: {score}")
    print(f"mean recall@5 example: {mean}")
