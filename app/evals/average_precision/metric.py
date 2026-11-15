"""Average precision over a ranked list, then the mean across queries."""


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        raise ValueError("relevant must not be empty")
    hits = 0
    total = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def mean_average_precision(cases: list[tuple[list[str], set[str]]]) -> float:
    if not cases:
        raise ValueError("cases must not be empty")
    return sum(average_precision(retrieved, relevant) for retrieved, relevant in cases) / len(cases)


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    score = average_precision(retrieved, relevant)
    mean = mean_average_precision([(retrieved, relevant), (["b"], relevant)])
    assert score == (1.0 + 2.0 / 3.0) / 2.0
    assert mean == score / 2.0
    print(f"average precision example: {score}")
    print(f"mean average precision example: {mean}")
