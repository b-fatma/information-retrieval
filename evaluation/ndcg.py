import math
from collections.abc import Collection, Sequence

from . import dcg_at_k


def ideal_dcg_at_k(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    k: int = 20,
) -> float:
    """Compute Ideal DCG@k (IDCG@k) using binary relevance.

    Relevance is 1 for relevant documents and 0 otherwise. The ideal ranking
    places all relevant documents at the top.

    Args:
        ranking: [(doc_id, score), ...] original ranking.
        relevant_docs: Collection of relevant document IDs.
        k: Cutoff position (default 20).

    Returns:
        IDCG@k value (max possible DCG at k using binary relevance).
    """
    if not ranking or not relevant_docs:
        return 0.0

    # Number of relevant docs
    num_relevant = sum(1 for doc_id, _ in ranking if doc_id in relevant_docs)

    # Ideal ranking: all relevant docs at the top, up to k
    ideal_rels = [1] * min(num_relevant, k)

    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        if rank == 1:
            idcg += rel
        else:
            idcg += rel / math.log2(rank)
    return idcg


def ndcg_at_k(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    k: int = 20,
) -> float:
    """Compute rank-based Normalized DCG@k.

    Uses binary relevance: 1 if the document is relevant, 0 otherwise.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.
        k: Cutoff position (default 20).

    Returns:
        NDCG@k value between 0 and 1.
    """
    if not ranking or not relevant_docs:
        return 0.0

    dcg = dcg_at_k(ranking, relevant_docs, k)
    idcg = ideal_dcg_at_k(ranking, relevant_docs, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg