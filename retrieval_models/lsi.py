import numpy as np

def lsi(inverted_index, processed_queries, k=100):
    """
    Latent Semantic Indexing (LSI) retrieval model.

    Args:
        inverted_index: {term: {doc_id: (tf, tfidf)}} (from CSV loader)
        processed_queries: {query_id: [terms]}
        k: number of latent dimensions

    Returns:
        similarities: {query_id: {doc_id: score}} sorted by descending score
        terms: list of terms corresponding to matrix rows
        docs: list of doc_ids corresponding to matrix columns
    """
    from .utils import build_weight_matrix

    # 1- Build term-document TF–IDF matrix
    W, terms, docs = build_weight_matrix(inverted_index)

    # 2- Compute SVD
    U, S, VT = np.linalg.svd(W, full_matrices=False)

    # 3- Reduce dimensionality
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    # 4- Build query vectors (binary weights)
    query_vectors = {}
    for qid, tokens in processed_queries.items():
        q_vec = np.array([1 if term in tokens else 0 for term in terms])
        query_vectors[qid] = q_vec

    # 5- Project queries into LSI space
    S_inv = np.linalg.inv(S_k)
    query_lsi = {}
    for qid, q_vec in query_vectors.items():
        query_lsi[qid] = q_vec.T @ U_k @ S_inv  # 1 x k vector

    # 6- Compute cosine similarity between query vectors and documents
    # Document vectors in latent space: S_k * VT_k (k x n_docs)
    doc_lsi = S_k @ VT_k
    similarities = {}
    for qid, q_vec in query_lsi.items():
        scores = q_vec @ doc_lsi  # 1 x n_docs
        sim_dict = {doc_id: float(score) for doc_id, score in zip(docs, scores.flatten())}
        # Sort descending
        similarities[qid] = dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))

    return similarities, terms, docs
