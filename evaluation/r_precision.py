from collections.abc import Collection, Sequence

from . import precision_at_k


def r_precision(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
) -> float:
    """R-Precision for a single query.

    R-Precision computes precision at rank R, where R is the number of
    relevant documents for the query. R-Precision = Precision@R.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        R-Precision value (between 0 and 1).
    """
    if not ranking:
        return 0.0

    R = len(relevant_docs)

    return precision_at_k(ranking, relevant_docs, R)