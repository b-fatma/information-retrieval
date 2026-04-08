def precision(ranking, relevant_docs):
    """
    Compute precision for a single query.

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs

    Returns:
        float: Precision value (between 0 and 1)
    """
    if not ranking:
        return 0.0

    # Extract only doc_ids
    ranked_doc_ids = [doc_id for doc_id, _ in ranking]

    # Count relevant retrieved
    rel_ret = sum(1 for doc_id in ranked_doc_ids if doc_id in relevant_docs)

    return rel_ret / len(ranked_doc_ids)

