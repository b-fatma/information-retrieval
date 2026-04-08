import numpy as np

def build_weight_matrix(inverted_index):
    """
    Build the term-document TF-IDF matrix from an inverted index.

    Args:
        inverted_index: {term: {doc_id: (tf, tfidf)}}

    Returns:
        W: numpy array of shape (num_terms, num_docs)
        terms: list of terms corresponding to rows of W
        docs: list of doc_ids corresponding to columns of W
    """
    terms = sorted(inverted_index.keys())
    docs = sorted({doc_id for doc_dict in inverted_index.values() for doc_id in doc_dict.keys()})

    W = np.zeros((len(terms), len(docs)))
    term_to_idx = {t: i for i, t in enumerate(terms)}
    doc_to_idx = {d: j for j, d in enumerate(docs)}

    for term, doc_dict in inverted_index.items():
        i = term_to_idx[term]
        for doc_id, stats in doc_dict.items():
            tfidf = stats[1] if isinstance(stats, tuple) else stats['tfidf']
            W[i, doc_to_idx[doc_id]] = tfidf

    return W, terms, docs


def build_binary_matrix(doc_term):
    """
    Build a binary term-document matrix (1 if term appears, 0 otherwise).

    Args:
        doc_term: {doc_id: {term: (tf, tfidf)}}

    Returns:
        matrix (np.ndarray): shape (num_terms, num_docs)
        terms: list of terms
        docs: list of document IDs
    """
    terms = sorted({term for t_info in doc_term.values() for term in t_info.keys()})
    docs = sorted(doc_term.keys())

    term_idx = {t: i for i, t in enumerate(terms)}
    doc_idx = {d: j for j, d in enumerate(docs)}
    matrix = np.zeros((len(terms), len(docs)))

    for doc_id, t_info in doc_term.items():
        j = doc_idx[doc_id]
        for term in t_info.keys():
            i = term_idx[term]
            matrix[i, j] = 1

    return matrix, terms, docs


def build_frequency_matrix(doc_term):
    """
    Build a frequency term-document matrix using term frequencies.

    Args:
        doc_term: {doc_id: {term: (tf, tfidf)}}

    Returns:
        matrix (np.ndarray): shape (num_terms, num_docs)
        terms: list of terms
        docs: list of document IDs
    """
    terms = sorted({term for t_info in doc_term.values() for term in t_info.keys()})
    docs = sorted(doc_term.keys())

    term_idx = {t: i for i, t in enumerate(terms)}
    doc_idx = {d: j for j, d in enumerate(docs)}
    matrix = np.zeros((len(terms), len(docs)))

    for doc_id, t_info in doc_term.items():
        j = doc_idx[doc_id]
        for term, (tf, _) in t_info.items():
            i = term_idx[term]
            matrix[i, j] = tf

    return matrix, terms, docs


def build_document_vocab(W, terms, docs):
    """
    Build a dictionary of term frequencies per document from a matrix.

    Args:
        W: term-document matrix
        terms: list of terms
        docs: list of doc_ids

    Returns:
        {doc_id: {term: tf}}
    """
    vocab = {}
    for j, doc in enumerate(docs):
        doc_vocab = {}
        for i, term in enumerate(terms):
            tf = W[i, j]
            if tf > 0:
                doc_vocab[term] = tf
        vocab[doc] = doc_vocab
    return vocab


def build_collection_vocab(doc_term=None, terms=None):
    """
    Return the collection vocabulary.

    Args:
        doc_term: optional, {doc_id: {term: (tf, tfidf)}}
        terms: optional, list of terms

    Returns:
        set of terms
    """
    if terms:
        return set(terms)
    elif doc_term:
        return {term for t_info in doc_term.values() for term in t_info.keys()}
    else:
        raise ValueError("Provide either doc_term or terms.")


def compute_collection_frequencies(W, terms):
    """
    Compute collection frequency for each term from a term-document matrix.

    Args:
        W: term-document matrix
        terms: list of terms

    Returns:
        {term: total_tf_in_collection}
    """
    return {term: np.sum(W[i, :]) for i, term in enumerate(terms)}


def compute_avgdl(doc_vocab):
    """
    Compute average document length (sum of term frequencies).

    Args:
        doc_vocab: {doc_id: {term: tf}}

    Returns:
        float
    """
    total_length = sum(sum(d.values()) for d in doc_vocab.values())
    return total_length / len(doc_vocab)


from .vsm import vsm
from .lsi import lsi
from .bir import bir
from .bm25 import bm25
from .language_models import (
    lm_mle, lm_laplace, lm_jelinek, lm_dirichlet
)

def run_model(
    model_name,
    inverted_index,
    doc_term,
    queries,
    relevance=None,
    params={},
    qid=None  # <-- optional single query ID
):
    """
    Runs a retrieval model and returns rankings:
    { qid: [(doc_id, score), ...] }
    """

    # If qid is provided, select only that query
    if qid is not None:
        queries = {qid: queries[qid]}
        if relevance is not None:
            relevance = {qid: relevance[qid]}

    if model_name == "vsm":
        return {
            qid: vsm(inverted_index, doc_term, q, sim=params.get("sim", "cosine"))
            for qid, q in queries.items()
        }

    elif model_name == "lsi":
        sims, _, _ = lsi(inverted_index, queries, k=params.get("k", 100))
        return {qid: list(sims[qid].items()) for qid in sims}

    elif model_name == "bir_binary_no_rel":
        return bir(
            inverted_index,
            doc_term,
            queries,
            relevance=None,
            mode="binary",
            use_relevance=False
        )

    elif model_name == "bir_binary_rel":
        return bir(
            inverted_index,
            doc_term,
            queries,
            relevance=relevance,
            mode="binary",
            use_relevance=True
        )

    elif model_name == "bir_extended_no_rel":
        return bir(
            inverted_index,
            doc_term,
            queries,
            relevance=None,
            mode="extended",
            use_relevance=False
        )

    elif model_name == "bir_extended_rel":
        return bir(
            inverted_index,
            doc_term,
            queries,
            relevance=relevance,
            mode="extended",
            use_relevance=True
        )

    elif model_name == "bm25":
        return bm25(
            inverted_index,
            doc_term,
            queries,
            k1=params.get("k1", 1.2),
            b=params.get("b", 0.75)
        )

    elif model_name == "lm_mle":
        return lm_mle(doc_term, queries)

    elif model_name == "lm_laplace":
        return lm_laplace(doc_term, queries)

    elif model_name == "lm_jelinek":
        return lm_jelinek(
            doc_term,
            queries,
            lam=params.get("lam", 0.2)
        )

    elif model_name == "lm_dirichlet":
        return lm_dirichlet(doc_term, queries)

    else:
        raise ValueError(f"Unknown model: {model_name}")
