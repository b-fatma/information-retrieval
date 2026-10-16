from collections.abc import Collection, Sequence


def recall(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
) -> float:
    """Compute recall for a single query.

    Recall (R) is the fraction of relevant documents that are retrieved.
    Recall = # relevant items retrieved / # relevant items.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        Recall value (between 0 and 1).
    """
    if not ranking or not relevant_docs:
        return 0.0

    # Extract only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in ranking]

    # Count relevant retrieved
    rel_ret = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs)

    return rel_ret / len(relevant_docs)