def reciprocal_rank(ranking, relevant_docs):
    """Compute reciprocal rank for a single query.

    Reciprocal Rank (RR) measures how early the first relevant document is retrieved. RR = 1 / rank of first relevant document

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs

    Returns:
        float: Reciprocal rank value (between 0 and 1)
    """
    if not ranking:
        return 0.0

    # Get rank of first relevant doc
    for idx, (doc_id, _) in enumerate(ranking):
        if doc_id in relevant_docs:
            r = idx + 1
            return 1 / r

    return 0.0