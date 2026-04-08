import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import precision
from helpers.load import (
    load_relevance_csv,
    load_doc_term_csv,
    load_inverted_index_csv,
    load_preprocessed_queries_csv
)
from retrieval_models import run_model

def main():
    # Load actual data
    doc_term = load_doc_term_csv('./data/doc_term.csv')
    inverted_index = load_inverted_index_csv('./data/inverted_index.csv')
    queries = load_preprocessed_queries_csv('./data/queries.csv')
    relevance = load_relevance_csv('./data/relevance.csv')

    # Run a model (LM Dirichlet as example)
    rankings = run_model(
        model_name="lm_dirichlet",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries
    )

    # Compute precision for the first query
    first_qid = list(queries.keys())[0]
    relevant_docs = relevance[first_qid]
    p = precision(rankings[first_qid], relevant_docs)
    print(f"Precision for Query {first_qid}: {p:.4f}")

    # Compute precision for all queries
    print("\nPrecision for all queries:")
    for qid, ranking in rankings.items():
        relevant_docs = relevance[qid]
        p = precision(ranking, relevant_docs)
        print(f"Query {qid}: {p:.4f}")

if __name__ == "__main__":
    main()