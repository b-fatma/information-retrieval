from collections.abc import Collection, Mapping, Sequence


def average_precision(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
) -> float:
    """Compute average precision for a single query.

    Average of the precision obtained at the rank positions where relevant
    documents are retrieved.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        Average precision value (between 0 and 1).
    """
    relevant = set(relevant_docs)

    if not ranking or not relevant:
        return 0.0

    hits = 0
    precisions: list[float] = []

    for k, (doc_id, _) in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / k)

    return sum(precisions) / len(precisions) if precisions else 0.0


def mean_average_precision(
    rankings: Mapping[int, Sequence[tuple[int, float]]],
    relevant_docs: Mapping[int, Collection[int]],
) -> float:
    """Compute mean average precision across multiple queries.

    Queries with no entry in `relevant_docs` are treated as having no
    relevant documents (AP = 0.0) and still count toward the mean.

    Args:
        rankings: Mapping of query IDs to ranked (doc_id, score) tuples,
            e.g., {1: [(doc_id, score), ...], 2: ...}.
        relevant_docs: Mapping of query IDs to relevant doc IDs,
            e.g., {1: {doc_id, ...}, 2: {doc_id, ...}}.

    Returns:
        Mean average precision value (between 0 and 1).
    """
    if not rankings:
        return 0.0

    avs = [
        average_precision(ranking, relevant_docs.get(qid, set()))
        for qid, ranking in rankings.items()
    ]
    return sum(avs) / len(avs)