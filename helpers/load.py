import csv
from collections import defaultdict


def load_doc_term_csv(filepath):
    """
    Load doc-term CSV: columns = ['doc_id', 'term', 'tf', 'tfidf']
    Returns:
        doc_term_dict: {doc_id: {term: (tf, tfidf)}}
    """
    doc_term_dict = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row['doc_id']
            term = row['term']
            tf = float(row['tf'])
            tfidf = float(row['tfidf'])
            doc_term_dict[doc_id][term] = (tf, tfidf)
    return dict(doc_term_dict)


def load_inverted_index_csv(filepath):
    """
    Load inverted index CSV: columns = ['term', 'doc_id', 'tf', 'tfidf']
    Returns:
        inverted_index: {term: {doc_id: (tf, tfidf)}}
    """
    inverted_index = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row['term']
            doc_id = row['doc_id']
            tf = float(row['tf'])
            tfidf = float(row['tfidf'])
            inverted_index[term][doc_id] = (tf, tfidf)
    return dict(inverted_index)


def load_preprocessed_queries_csv(filepath):
    """
    Load preprocessed queries CSV: columns = ['query_id', 'terms']
    Returns:
        queries: {query_id: [terms]}
    """
    queries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row['query_id'])
            terms = row['terms'].split()
            queries[qid] = terms
    return queries


def load_relevance_csv(filepath):
    """
    Load relevance CSV: columns = ['query_id', 'doc_id']
    Returns:
        rels: {query_id: set(doc_ids)}
    """
    rels = defaultdict(set)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row['query_id'])
            doc_id = row['doc_id']
            rels[qid].add(doc_id)
    return dict(rels)