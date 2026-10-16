from . import interpolated_precision_recall_curve

def interpolated_average_precision(ranking, relevant_docs):
    """Compute the interpolated average precision for a single query.

    This calculates the average precision using interpolated precision values 
    at fixed recall thresholds (typically the standard 11-point interpolation 
    from 0.0 to 1.0 with a step of 0.1).

    Args:
        ranking (list of tuples): [(doc_id, score), ...] in ranked order.
        relevant_docs (collection): Set, list, or dict of relevant document IDs.

    Returns:
        float: The interpolated average precision value (between 0.0 and 1.0).
    """
    interpolated_precision_values = interpolated_precision_recall_curve(ranking, relevant_docs)["precision"]
    
    if not interpolated_precision_values:
        return 0.0
        
    return sum(interpolated_precision_values) / len(interpolated_precision_values) 


def interpolated_map(rankings, relevant_docs):
    """Compute the Mean Interpolated Average Precision across multiple queries.

    This calculates the mean over all system queries of their individual 
    interpolated average precision scores.

    Args:
        rankings (dict): Mapping of query IDs to ranked document tuples,
            e.g., {"Q1": [(doc_id, score), ...], ...}.
        relevant_docs (dict): Mapping of query IDs to their ground-truth 
            relevant document collections, e.g., {"Q1": {doc_id, ...}, ...}.

    Returns:
        float: The mean interpolated average precision score (between 0.0 and 1.0).
    """
    if not rankings:
        return 0.0

    iavs = []
    for qid, ranking in rankings.items():
        q_relevant = relevant_docs.get(qid, set())
        iavs.append(interpolated_average_precision(ranking, q_relevant))
        
    return sum(iavs) / len(iavs)
