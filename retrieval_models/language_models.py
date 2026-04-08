import numpy as np
from .utils import (
    build_document_vocab, 
    build_collection_vocab, 
    compute_collection_frequencies, 
    build_frequency_matrix
)

# Core LM Scoring Functions
def _mle_score(query_terms, doc_vocab):
    N_d = sum(doc_vocab.values())
    if N_d == 0:
        return -np.inf
    log_prob = 0.0
    for w in query_terms:
        tf = doc_vocab.get(w, 0)
        if tf == 0:
            return -np.inf
        log_prob += np.log10(tf / N_d)
    return log_prob

def _laplace_score(query_terms, doc_vocab, collection_vocab, k=1):
    N_d = sum(doc_vocab.values()) + k * len(collection_vocab)
    log_prob = 0.0
    for w in query_terms:
        tf = doc_vocab.get(w, 0) + k
        log_prob += np.log10(tf / N_d)
    return log_prob

def _jelinek_mercer_score(query_terms, doc_vocab, cf, lam=0.4):
    N_d = sum(doc_vocab.values())
    N_c = sum(cf.values())
    log_prob = 0.0
    for w in query_terms:
        p_doc = doc_vocab.get(w, 0) / N_d if N_d > 0 else 0
        p_coll = cf.get(w, 0) / N_c if N_c > 0 else 0
        p = lam * p_doc + (1 - lam) * p_coll
        if p > 0:
            log_prob += np.log10(p)
        else:
            log_prob += -np.inf
    return log_prob

def _dirichlet_score(query_terms, doc_vocab, cf, mu=0.3):
    N_d = sum(doc_vocab.values())
    N_c = sum(cf.values())
    log_prob = 0.0
    for w in query_terms:
        p_coll = cf.get(w, 0) / N_c if N_c > 0 else 0
        p = (doc_vocab.get(w, 0) + mu * p_coll) / (N_d + mu)
        if p > 0:
            log_prob += np.log10(p)
        else:
            log_prob += -np.inf
    return log_prob


# Main LM Interface Functions


def lm_mle(doc_term, queries):
    W, terms, docs = build_frequency_matrix(doc_term)
    doc_vocab = build_document_vocab(W, terms, docs)

    rankings = {}
    for qid, tokens in queries.items():
        scores = {doc_id: _mle_score(tokens, dv) for doc_id, dv in doc_vocab.items()}
        rankings[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return rankings


def lm_laplace(doc_term, queries, k=1):
    W, terms, docs = build_frequency_matrix(doc_term)
    doc_vocab = build_document_vocab(W, terms, docs)
    collection_vocab = build_collection_vocab(terms=terms)

    rankings = {}
    for qid, tokens in queries.items():
        scores = {doc_id: _laplace_score(tokens, dv, collection_vocab, k) for doc_id, dv in doc_vocab.items()}
        rankings[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return rankings


def lm_jelinek(doc_term, queries, lam=0.2):
    W, terms, docs = build_frequency_matrix(doc_term)
    doc_vocab = build_document_vocab(W, terms, docs)
    cf = compute_collection_frequencies(W, terms)

    rankings = {}
    for qid, tokens in queries.items():
        scores = {doc_id: _jelinek_mercer_score(tokens, dv, cf, lam) for doc_id, dv in doc_vocab.items()}
        rankings[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return rankings


def lm_dirichlet(doc_term, queries, mu=0.3):
    W, terms, docs = build_frequency_matrix(doc_term)
    doc_vocab = build_document_vocab(W, terms, docs)
    cf = compute_collection_frequencies(W, terms)

    # Scale mu by average document length
    avgdl = np.mean([sum(dv.values()) for dv in doc_vocab.values()])
    mu_scaled = mu * avgdl

    rankings = {}
    for qid, tokens in queries.items():
        scores = {doc_id: _dirichlet_score(tokens, dv, cf, mu_scaled) for doc_id, dv in doc_vocab.items()}
        rankings[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return rankings
