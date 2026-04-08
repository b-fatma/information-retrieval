from helpers.load import *
from retrieval_models import run_model


def print_top10(model_name, rankings, qid):
    print(f"\nTesting {model_name} on Query {qid}")
    print(f"{'Rank':<5} {'DocID':<10} {'Score':<15}")
    print("-" * 35)

    for rank, (doc_id, score) in enumerate(rankings[qid][:10], start=1):
        print(f"{rank:<5} {doc_id:<10} {score:<15.6f}")


if __name__ == "__main__":
    # ----------------------
    # Load data
    # ----------------------
    doc_term = load_doc_term_csv('./data/doc_term.csv')
    inverted_index = load_inverted_index_csv('./data/inverted_index.csv')
    queries = load_preprocessed_queries_csv('./data/queries.csv')
    relevance = load_relevance_csv('./data/relevance.csv')

    print(f"Loaded {len(doc_term)} documents")
    print(f"Loaded {len(inverted_index)} terms in inverted index")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded relevance judgments for {len(relevance)} queries")

    # ----------------------
    # Use first query only
    # ----------------------
    first_qid = list(queries.keys())[0]

    # ----------------------
    # VSM
    # ----------------------
    rankings = run_model(
        model_name="vsm",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        params={"sim": "cosine"},
        qid=first_qid
    )
    print_top10("VSM", rankings, first_qid)

    # ----------------------
    # LSI
    # ----------------------
    rankings = run_model(
        model_name="lsi",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        params={"k": 100},
        qid=first_qid
    )
    print_top10("LSI", rankings, first_qid)

    # ----------------------
    # BIR Binary (No Relevance)
    # ----------------------
    rankings = run_model(
        model_name="bir_binary_no_rel",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        qid=first_qid
    )
    print_top10("BIR Binary (No Rel)", rankings, first_qid)

    # ----------------------
    # BIR Extended (No Relevance)
    # ----------------------
    rankings = run_model(
        model_name="bir_extended_no_rel",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        qid=first_qid
    )
    print_top10("BIR Extended (No Rel)", rankings, first_qid)

    # ----------------------
    # BIR Binary (With Relevance)
    # ----------------------
    rankings = run_model(
        model_name="bir_binary_rel",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        relevance=relevance,
        qid=first_qid
    )
    print_top10("BIR Binary (Rel)", rankings, first_qid)

    # ----------------------
    # BIR Extended (With Relevance)
    # ----------------------
    rankings = run_model(
        model_name="bir_extended_rel",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        relevance=relevance,
        qid=first_qid
    )
    print_top10("BIR Extended (Rel)", rankings, first_qid)

    # ----------------------
    # BM25
    # ----------------------
    rankings = run_model(
        model_name="bm25",
        inverted_index=inverted_index,
        doc_term=doc_term,
        queries=queries,
        params={"k1": 1.5, "b": 0.75},
        qid=first_qid
    )
    print_top10("BM25", rankings, first_qid)

    # ----------------------
    # Language Models
    # ----------------------
    rankings = run_model(
        "lm_mle",
        inverted_index,
        doc_term,
        queries,
        qid=first_qid
    )
    print_top10("LM MLE", rankings, first_qid)

    rankings = run_model(
        "lm_laplace",
        inverted_index,
        doc_term,
        queries,
        qid=first_qid
    )
    print_top10("LM Laplace", rankings, first_qid)

    rankings = run_model(
        "lm_jelinek",
        inverted_index,
        doc_term,
        queries,
        params={"lam": 0.2},
        qid=first_qid
    )
    print_top10("LM Jelinek-Mercer", rankings, first_qid)

    rankings = run_model(
        "lm_dirichlet",
        inverted_index,
        doc_term,
        queries,
        qid=first_qid
    )
    print(rankings)
    print_top10("LM Dirichlet", rankings, first_qid)
