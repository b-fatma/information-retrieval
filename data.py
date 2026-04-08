from helpers.parse import *
from helpers.preprocess import *
from helpers.save import *

if __name__ == "__main__":
    output_path = 'data'
    os.makedirs(output_path, exist_ok=True)

    # Parse MEDLINE collection
    docs = parse_medline_docs('med/MED.ALL')
    assert len(docs) == 1033 # ensure correspondance with length provided in assignment

    queries = parse_medline_queries('med/MED.QRY')
    assert len(queries) == 30
    rels = parse_medline_rel('med/MED.REL')

    # Preprocess documents and queries
    pre_docs = preprocess_collection(docs)
    pre_queries = preprocess_collection(queries)

    # Term frequencies and TF-IDF for documents
    term_freqs = compute_term_freq(pre_docs)
    df = compute_doc_frequencies(term_freqs)
    N = len(docs)
    tfidf_docs = compute_tfidf(term_freqs, df, N)

    # Save CSVs
    save_doc_term_csv(os.path.join(output_path, 'doc_term.csv'), term_freqs, tfidf_docs)
    save_inverted_index_csv(os.path.join(output_path, 'inverted_index.csv'), term_freqs, tfidf_docs)
    save_queries_csv(os.path.join(output_path, 'queries.csv'), pre_queries)
    save_relevance_csv(os.path.join(output_path, 'relevance.csv'), rels)
