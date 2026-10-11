from . import precision_recall_curve
import numpy as np

def interpolated_precision_recall_curve(ranking, relevant_docs, recall_levels=None):
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
