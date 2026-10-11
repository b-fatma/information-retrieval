def average_precision(ranking, relevant_docs):
    """Compute average precision for a single query.

    Average of the precision obtained at the rank positions where relevant documents are retrieved.

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order
        relevant_docs (set or list): Set or list of relevant document IDs
    
    Returns:
        float: Average precision value (between 0 and 1)
    """

    relevant = set(relevant_docs)

    if not ranking or not relevant:
        return 0.0

    hits = 0
    precisions = list()

    for k, (doc_id, _) in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / k)

    return sum(precisions) / len(precisions)


def mean_average_precision(rankings, relevant_docs):
    """Compute mean average precision.

    Args:
        rankings (dict): {"Q1": [(doc_id, score), ...], "Q2": ...}
        relevant_docs (dict): {"Q1": {doc_id, ...}, "Q2": {doc_id, ...}}
    
    Returns:
        float: Mean average precision value (between 0 and 1)
    """
    if not rankings:
        return 0.0

    avs = []
    for qid, ranking in rankings.items():
        q_relevant = relevant_docs.get(qid, set())
        avs.append(average_precision(ranking, q_relevant))
        
    return sum(avs) / len(avs)