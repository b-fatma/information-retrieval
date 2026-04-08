from helpers.load import *

if __name__ == "__main__":
    doc_term = load_doc_term_csv('./data/doc_term.csv')
    inverted_index = load_inverted_index_csv('./data/inverted_index.csv')
    queries = load_preprocessed_queries_csv('./data/queries.csv')
    relevance = load_relevance_csv('./data/relevance.csv')

    print(f"Loaded {len(doc_term)} documents")
    print(f"Loaded {len(inverted_index)} terms in inverted index")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded relevance judgments for {len(relevance)} queries")
