import math
from . import dcg_at_k

def ideal_dcg_at_k(ranking, relevant_docs, k=20):
    """
    Compute Ideal DCG@k (IDCG@k) using rank-based relevance (1 for relevant, 0 for non-relevant).
    
    Args:
        ranking (list of tuples): [(doc_id, score), ...] original ranking
        relevant_docs (set or list): Set of relevant document IDs
        k (int): Cutoff position (default=20)
    
    Returns:
        float: IDCG@k value (max possible DCG at k using rank-based relevance)
    """
    if not ranking or not relevant_docs:
        return 0.0
    
    # Number of relevant docs
    num_relevant = sum(1 for doc_id, _ in ranking if doc_id in relevant_docs)
    
    # Ideal ranking: all relevant docs at the top
    ideal_rels = [1] * min(num_relevant, k)  # 1 for each relevant doc, up to k
    
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        if rank == 1:
            idcg += rel
        else:
            idcg += rel / math.log2(rank)
    
    return idcg


def ndcg_at_k(ranking, relevant_docs, k=20):
    """
    Compute rank-based Normalized DCG@k.
    
    Uses binary relevance: 1 if document is relevant, 0 otherwise.
    
    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set of relevant document IDs
        k (int): Cutoff position (default=20)
    
    Returns:
        float: NDCG@k value between 0 and 1
    """
    if not ranking or not relevant_docs:
        return 0.0
    
    dcg = dcg_at_k(ranking, relevant_docs, k)
    
    # Calculate ideal DCG
    idcg = ideal_dcg_at_k(ranking, relevant_docs, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg