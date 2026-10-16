from collections.abc import Collection, Sequence


def recall_at_k(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    k: int,
) -> float:
    """Compute recall@k for a single query.

    Recall@K (R@K) measures the proportion of relevant documents that are
    among the top-K retrieved results.
    R@K = # relevant documents in top-K / # relevant documents.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.
        k: Cutoff position.

    Returns:
        Recall@k value (between 0 and 1).
    """
    if not relevant_docs or not ranking or k <= 0:
        return 0.0

    # If k is larger than the number of docs
    k = min(k, len(ranking))

    # Extract only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in ranking[:k]]

    # Count relevant retrieved
    rel_ret = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs)

    return rel_ret / len(relevant_docs)