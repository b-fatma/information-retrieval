import os
import csv

def save_doc_term_csv(filepath, term_freqs, tfidf):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'term', 'tf', 'tfidf'])
        for doc_id in term_freqs:
            for term in term_freqs[doc_id]:
                writer.writerow([doc_id, term, term_freqs[doc_id][term], tfidf[doc_id][term]])


def save_inverted_index_csv(filepath, term_freqs, tfidf):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['term', 'doc_id', 'tf', 'tfidf'])
        for doc_id in term_freqs:
            for term in term_freqs[doc_id]:
                writer.writerow([term, doc_id, term_freqs[doc_id][term], tfidf[doc_id][term]])


def save_relevance_csv(filepath, rels):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'doc_id'])
        for qid, doc_ids in rels.items():
            for doc_id in doc_ids:
                writer.writerow([qid, doc_id])


def save_queries_csv(filepath, pre_queries):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'terms'])  # header
        for qid, terms in pre_queries.items():
            writer.writerow([qid, ' '.join(terms)])