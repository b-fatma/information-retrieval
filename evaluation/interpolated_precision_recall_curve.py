from collections.abc import Collection, Mapping, Sequence

from . import precision_recall_curve
import numpy as np

def interpolated_precision_recall_curvedef(
    ranking: Sequence[tuple[int, float]],
    relevant_docs: Collection[int],
    recall_levels: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Compute the interpolated precision-recall curve for a single query.

    For each recall level r, the interpolated precision is the maximum
    precision observed at any point on the raw precision-recall curve where
    recall >= r. This produces a monotonically non-increasing curve.

    Args:
        ranking: [(doc_id, score), ...] in ranked order.
        relevant_docs: Collection of relevant document IDs.
        recall_levels: Recall thresholds at which to evaluate the curve.
            Defaults to the standard 11 points (0.0, 0.1, ..., 1.0).

    Returns:
        A dict with two keys:
            "precision": interpolated precision at each recall level.
            "recall": the recall levels used (same length as "precision").
    """
    if recall_levels is None:
        recall_levels = list(np.arange(0.0, 1.1, 0.1))
    
    interpolated_curve = {"precision": [], "recall": recall_levels}
    
    if not ranking or not relevant_docs:
        return {"precision": [0.0] * len(recall_levels), "recall": recall_levels}

    curve = precision_recall_curve(ranking, relevant_docs)
    precision_points = curve["precision"]
    recall_points = curve["recall"]

    for recall in recall_levels:
        # Find the maximum precision for any point where recall_points[k] >= recall
        max_p = 0.0
        for p, r in zip(precision_points, recall_points):
            if r >= recall and p > max_p:
                max_p = p
        interpolated_curve["precision"].append(max_p)

    return interpolated_curve
