import numpy as np

def vsm(inverted_index, doc_term, processed_query, sim='cosine'):
    """
    Vector Space Model using cosine similarity.

    Args:
        inverted_index: {term: {doc_id: (tf, tfidf)}}
        doc_term: {doc_id: {term: (tf, tfidf)}}
        processed_query: list of preprocessed query terms
        sim: similarity measure ('cosine' or 'inner_product')

    Returns:
        List of tuples [(doc_id, score), ...] sorted descending by score
    """
    RSVs = {}

    # Query weights (binary)
    query_weights = {term: 1 for term in processed_query}
    norm_q = np.sqrt(sum(w * w for w in query_weights.values()))

    for doc_id in doc_term.keys():
        inner_product = 0.0

        # Dot product between query and document
        for term in processed_query:
            if term in inverted_index and doc_id in inverted_index[term]:
                doc_weight = inverted_index[term][doc_id][1]  # tfidf
                query_weight = query_weights[term]
                inner_product += doc_weight * query_weight

        if sim == 'inner_product':
            RSVs[doc_id] = inner_product
        else:
            # Document norm
            doc_weights = [info[1] ** 2 for info in doc_term[doc_id].values()]  # tfidf
            norm_doc = np.sqrt(np.sum(doc_weights))

            if norm_doc == 0 or norm_q == 0:
                score = 0
            elif sim == 'cosine':
                score = inner_product / (norm_q * norm_doc)
            else:
                raise ValueError(f"Unknown similarity measure: {sim}")

            RSVs[doc_id] = score

    # Sort by descending score
    RSVs = sorted(RSVs.items(), key=lambda x: x[1], reverse=True)
    return RSVs
