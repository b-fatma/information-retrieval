def precision_recall_curve(ranking, relevant_docs):
    """Compute coordinates for the precision-recall curve.

    Precision@k and recall@k are computed at each rank k where a relevant document is retrieved.

    Args:
        ranking: [(doc_id, score), ...] in ranked order, no duplicate doc_ids.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        {"precision": [...], "recall": [...]}; lists are empty if there is no ranking or no relevant documents.
    """
    relevant = set(relevant_docs)
    curve = {"precision": [], "recall": []}

    if not ranking or not relevant:
        return curve

    n_relevant = len(relevant)
    hits = 0

    for k, (doc_id, _) in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            curve["precision"].append(hits / k)
            curve["recall"].append(hits / n_relevant)

    return curve