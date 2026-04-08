import numpy as np
from .utils import build_binary_matrix, build_frequency_matrix

def bm25(inverted_index, doc_term, queries, k1=1.2, b=0.75):
    """
    BM25 retrieval model.

    Args:
        inverted_index: {term: {doc_id: (tf, tfidf)}}
        doc_term: {doc_id: {term: (tf, tfidf)}}
        queries: {qid: [tokens]}  # preprocessed queries
        k1: BM25 term frequency parameter
        b: BM25 length normalization parameter

    Returns:
        rankings: {qid: [(doc_id, score), ...]} sorted descending by score
    """
    terms = sorted(inverted_index.keys())
    docs = sorted(doc_term.keys())

    # Build matrices
    B, _, _ = build_binary_matrix(doc_term)
    tf_matrix, _, _ = build_frequency_matrix(doc_term)

    # Document lengths and average document length
    dls = B.sum(axis=0)
    avdl = np.mean(dls)

    rankings = {}

    for qid, tokens in queries.items():
        # Indices of query terms in the terms list
        idx = [terms.index(t) for t in tokens if t in terms]
        N = B.shape[1]
        ns = B.sum(axis=1)
        scores = np.zeros(N)

        for i in idx:
            idf = np.log10((N - ns[i] + 0.5) / (ns[i] + 0.5))
            tf = tf_matrix[i, :]
            scores += ((k1 + 1) * tf) / (k1 * (1 - b + b * (dls / avdl)) + tf) * idf

        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        rankings[qid] = ranked_docs

    return rankings



