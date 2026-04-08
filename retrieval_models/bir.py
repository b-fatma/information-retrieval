import numpy as np
from .utils import build_binary_matrix, build_weight_matrix

def bir(inverted_index, doc_term, queries, relevance=None, mode="binary", use_relevance=False):
    """
    Binary Independence Retrieval (BIR) model: Classic or Extended, with or without relevance feedback.

    Args:
        inverted_index: {term: {doc_id: (tf, tfidf)}} 
        doc_term: {doc_id: {term: (tf, tfidf)}}
        queries: {qid: [tokens]}  # preprocessed queries
        relevance: {qid: set(doc_ids)} (optional)
        mode: "binary" or "extended"
        use_relevance: True/False, whether to use relevance feedback

    Returns:
        rankings: {qid: [(doc_id, score), ...]} sorted descending by score
    """
    terms = sorted(inverted_index.keys())
    docs = sorted(doc_term.keys())

    # Build document-term matrices
    B, _, _ = build_binary_matrix(doc_term)
    W = None
    if mode == "extended":
        W, _, _ = build_weight_matrix(inverted_index)  # actual tfidf weights

    rankings = {}

    for qid, tokens in queries.items():
        relevant_docs = relevance.get(qid) if (use_relevance and relevance) else None
        indices = [terms.index(t) for t in tokens if t in terms]
        N = len(docs)
        scores = np.zeros(N)

        if relevant_docs is None:
            # Classic or Extended without relevance
            ns = B.sum(axis=1)  # always use binary counts for ns
            for idx in indices:
                idf = np.log10((N - ns[idx] + 0.5) / (ns[idx] + 0.5))
                if mode == "binary":
                    scores += B[idx, :] * idf
                else:  # extended
                    scores += W[idx, :] * idf
        else:
            # With relevance feedback
            R = len(relevant_docs)
            rel_set = set(relevant_docs)
            doc_index = {d: i for i, d in enumerate(docs)}
            ns = B.sum(axis=1)
            for idx in indices:
                term_row = W[idx, :] if mode == "extended" else B[idx, :]
                r_i = np.sum([B[idx, doc_index[d]] > 0 for d in rel_set])
                n_i = ns[idx]
                numerator = (r_i + 0.5) / (R - r_i + 0.5)
                denominator = (n_i - r_i + 0.5) / (N - n_i - R + r_i + 0.5)
                w_i = np.log10(numerator / denominator)
                scores += term_row * w_i

        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        rankings[qid] = ranked_docs

    return rankings
