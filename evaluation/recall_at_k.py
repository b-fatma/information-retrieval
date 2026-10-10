def recall_at_k(ranking, relevant_docs, k: int):
    """
    Compute recall@k for a single query.

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs
        k (int): position

    Returns:
        float: Recall@k value (between 0 and 1)
    """
    
    if not relevant_docs:
        return 0.0

    if not ranking or k <= 0:
        return 0.0

    # If k is larger than the number of docs
    k = min(k, len(ranking))

    # Extract only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in ranking[:k]]

    # Count relevant retrieved
    rel_ret = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs)

    
    return rel_ret / len(relevant_docs)