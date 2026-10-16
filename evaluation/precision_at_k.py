from collections.abc import Collection, Sequence


def precision_at_k(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    k: int,
) -> float:
    """Compute precision@k for a single query.

    Precision@K (P@K) measures the proportion of relevant documents among the
    top-K retrieved results. P@K = # relevant documents in top-K / K.

    If the ranking contains fewer than K documents, the denominator is the
    number of documents actually retrieved.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.
        k: Cutoff position.

    Returns:
        Precision@k value (between 0 and 1).
    """
    if not ranking or k <= 0:
        return 0.0

    # If k is larger than the number of docs
    denom = min(k, len(ranking))

    # Extract only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in ranking[:denom]]

    # Count relevant retrieved
    rel_ret = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs)

    return rel_ret / denom