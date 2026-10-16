from collections.abc import Collection, Mapping, Sequence

from . import interpolated_precision_recall_curve


def interpolated_average_precision(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
) -> float:
    """Compute the interpolated average precision for a single query.

    This calculates the average precision using interpolated precision values
    at fixed recall thresholds (typically the standard 11-point interpolation
    from 0.0 to 1.0 with a step of 0.1).

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.

    Returns:
        The interpolated average precision value (between 0.0 and 1.0).
    """
    interpolated_precision_values = interpolated_precision_recall_curve(
        ranking, relevant_docs
    )["precision"]

    if not interpolated_precision_values:
        return 0.0

    return sum(interpolated_precision_values) / len(interpolated_precision_values)


def interpolated_map(
    rankings: Mapping[int, Sequence[tuple[int, float]]],
    relevant_docs: Mapping[int, Collection[int]],
) -> float:
    """Compute the Mean Interpolated Average Precision across multiple queries.

    This calculates the mean over all system queries of their individual
    interpolated average precision scores. Queries with no entry in
    `relevant_docs` are treated as having no relevant documents (score 0.0)
    and still count toward the mean.

    Args:
        rankings: Mapping of query IDs to ranked (doc_id, score) tuples,
            e.g., {1: [(doc_id, score), ...], ...}.
        relevant_docs: Mapping of query IDs to their ground-truth relevant
            document collections, e.g., {1: {doc_id, ...}, ...}.

    Returns:
        The mean interpolated average precision score (between 0.0 and 1.0).
    """
    if not rankings:
        return 0.0

    iavs = [
        interpolated_average_precision(ranking, relevant_docs.get(qid, set()))
        for qid, ranking in rankings.items()
    ]
    return sum(iavs) / len(iavs)