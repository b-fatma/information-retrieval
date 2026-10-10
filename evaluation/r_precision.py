def r_precision(ranking, relevant_docs):
    """R-Precision for a single query.

    R-Precision is equivalent to Precision@R

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs

    Returns:
        float: R-Precision value (between 0 and 1)
    """
    if not ranking:
        return 0.0

    R = len(relevant_docs)

    from . import precision_at_k

    return precision_at_k(ranking, relevant_docs, R)