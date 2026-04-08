#!/bin/bash

# Create evaluation directory
mkdir -p ../evaluation

# List of metric files 
metrics=(
    "precision"
    "recall"
    "f1"
    "precision_recall_curve"
    "interpolated_precision_recall_curve"
    "map"
    "interpolated_map"
    "precision_at_k"
    "r_precision"
    "rr"
    "dcg"
    "ndcg"
    "gain"
    "utils"
)

# Create each metric file with a placeholder
for metric in "${metrics[@]}"; do
    touch "../evaluation/${metric}.py"
    echo "# TODO: Implement ${metric} metric functions" > "../evaluation/${metric}.py"
done

# Create __init__.py with the specified imports
cat <<EOL > ../evaluation/__init__.py
from .precision import precision, precision_at_k
from .recall import recall
from .f1 import f1_score
from .precision_recall_curve import precision_recall_curve
from .interpolated_precision_recall_curve import interpolated_precision_recall_curve
from .map import mean_average_precision
from .interpolated_map import interpolated_map
from .r_precision import r_precision
from .rr import reciprocal_rank
from .dcg import dcg_at_k
from .ndcg import ndcg_at_k
from .gain import gain
from .utils import *
EOL

echo "Evaluation folder and all required files created successfully!"
