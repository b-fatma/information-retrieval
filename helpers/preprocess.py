import os
import numpy as np
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import nltk


nltk.download('stopwords')

tokenizer = RegexpTokenizer(
    r'(?:[A-Za-z]\.)+'                     
    r'|[A-Za-z]+[\-@]\d+(?:\.\d+)?'       
    r'|\d+(?:[\.\,\-]\d+)*%?'             
    r'|[A-Za-z]+')                         

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [porter.stem(t) for t in tokens]
    return tokens


def preprocess_collection(collection):
    return {doc_id: preprocess_text(text) for doc_id, text in collection.items()}

# ----------------------------
# Term frequency and TF-IDF
# ----------------------------
def compute_term_freq(preprocessed_docs):
    term_freqs = {}
    for doc_id, tokens in preprocessed_docs.items():
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        term_freqs[doc_id] = dict(tf)
    return term_freqs


def compute_doc_frequencies(term_freqs):
    df = defaultdict(int)
    for doc_tf in term_freqs.values():
        for term in doc_tf.keys():
            df[term] += 1
    return dict(df)


def compute_tfidf(term_freqs, df, N):
    tfidf = {}
    for doc_id, doc_tf in term_freqs.items():
        tfidf[doc_id] = {}
        max_freq = max(doc_tf.values())
        for term, freq in doc_tf.items():
            tf = freq / max_freq
            idf = np.log10(N / df[term] + 1)
            tfidf[doc_id][term] = float(tf * idf)
    return tfidf