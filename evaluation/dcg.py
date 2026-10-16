import math
from collections.abc import Collection, Sequence

def dcg_at_k(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    k: int =20
) -> float:
    """Compute Discounted Cumulative Gain at k for a single query.
    
    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs
        k (int): Cutoff position (default=20)
    
    Returns:
        float: DCG@k value
    """
    if not ranking:
        return 0.0
    
    dcg = 0.0
    
    for rank, (doc_id, score) in enumerate(ranking[:k], start=1):
        ri = 1 if doc_id in relevant_docs else 0
        if rank == 1:  # Position 1
            dcg += ri
        else:  # Position 2 to k
            dcg += ri / math.log2(rank)
    
    return dcg