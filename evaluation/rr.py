from collections.abc import Collection, Sequence


def reciprocal_rank(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
) -> float:
    """Compute reciprocal rank for a single query.

    Reciprocal Rank (RR) measures how early the first relevant document is
    retrieved. RR = 1 / rank of first relevant document. Returns 0.0 if no
    relevant document is retrieved.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        Reciprocal rank value (between 0 and 1).
    """
    if not ranking:
        return 0.0

    # Get rank of first relevant doc
    for rank, (doc_id, _) in enumerate(ranking, start=1):
        if doc_id in relevant_docs:
            return 1 / rank

    return 0.0