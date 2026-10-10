def f1_score(ranking, relevant_docs):
    """Compute f1-score for one query.

    f1-score = (2 * precision * recall) / (precision + recall)

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs

    Returns:
        float: f1-score value (between 0 and 1)
    """

    from . import precision, recall

    p = precision(ranking, relevant_docs)
    r = recall(ranking, relevant_docs)

    return 2 * p * r / (p + r)