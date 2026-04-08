# Information Retrieval Project

This repository implements various **Information Retrieval (IR) models** and evaluates them on the **MEDLINE dataset**. It provides preprocessing, indexing, retrieval, and evaluation metrics in a modular framework.

---

## 📁 Project Structure

```text
.
├── data/                    # Preprocessed CSVs: doc-term matrix, queries, relevance
├── evaluation/              # Metrics implementations (Precision)
├── helpers/                 # Loading and preprocessing utilities
├── med/                     # Original MEDLINE dataset files
├── retrieval_models/        # Implemented IR models (VSM, BM25, LSI, BIR, Language Models)
├── test_metrics/            # Tests for metrics
├── test_load.py             # Tests for data loading
├── test_models.py           # Tests for IR models
├── data.py                  # Preprocessing, term weighting, and indexing
└── README.md                # This file
```

---

## 📝 Dataset

* **Documents:** MEDLINE abstracts (`MED.ALL`)
* **Queries:** Provided in the MEDLINE dataset (`MED.QRY`)
* **Relevance judgments:** Binary relevance (`MED.REL`)

**Note:** The queries are already part of the dataset and are preprocessed before running retrieval models.

---

## 🔹 Preprocessing Pipeline

1. **Tokenization:** Regex-based, capturing words, abbreviations, numbers, and alphanumerics.
2. **Stopword Removal:** English stopwords removed.
3. **Stemming:** Porter stemming applied.
4. **Term Weighting:**

   * TF (Term Frequency)
   * TF-IDF (log-normalized)

All steps are implemented in `data.py`.

---

## 🔹 Indexing

* **Document-Term Matrix:** Stored in `data/doc_term.csv`.
* **Inverted Index:** Maps terms to document occurrences (`data/inverted_index.csv`).

---

## 🔹 Retrieval Models

| Model               | Description                               |
| ------------------- | ----------------------------------------- |
| **VSM**             | Vector Space Model with cosine similarity |
| **LSI**             | Latent Semantic Indexing (SVD)            |
| **BIR**             | Binary Independence Model                 |
| **BM25**            | Okapi BM25 probabilistic ranking          |
| **Language Models** | MLE, Laplace, Jelinek-Mercer, Dirichlet   |

All models can be run via the unified `run_model` function.

---

## 🔹 Usage Example

```python
from helpers.load import (
    load_doc_term_csv,
    load_inverted_index_csv,
    load_preprocessed_queries_csv,
    load_relevance_csv
)
from retrieval_models import run_model
from evaluation import precision

# Load data
doc_term = load_doc_term_csv('./data/doc_term.csv')
inverted_index = load_inverted_index_csv('./data/inverted_index.csv')
queries = load_preprocessed_queries_csv('./data/queries.csv')
relevance = load_relevance_csv('./data/relevance.csv')

# Run Language Model (Dirichlet) as an example
rankings = run_model(
    model_name="lm_dirichlet",
    inverted_index=inverted_index,
    doc_term=doc_term,
    queries=queries
)

# Compute precision for the first query
first_qid = list(queries.keys())[0]
p = precision(rankings[first_qid], relevance[first_qid])
print(f"Precision for Query {first_qid}: {p:.4f}")

# Compute precision for all queries
for qid, ranking in rankings.items():
    p = precision(ranking, relevance[qid])
    print(f"Query {qid}: {p:.4f}")
```

---

## 🔹 Evaluation Metrics

Currently implemented:

* **Precision:** Fraction of retrieved documents that are relevant.
* Additional metrics can be added modularly (Recall, F1, MAP, nDCG).

---

## 🔹 Testing

Run all tests with:

```bash
pytest
```

* `test_metrics/` — tests for evaluation metrics
* `test_load.py` — tests for data loading
* `test_models.py` — tests for retrieval models

---

## 📌 Notes

* MEDLINE queries are included in the dataset (`MED.QRY`) and preprocessed before retrieval.
* Relevance judgments are binary.
* Modular design allows adding new models and metrics easily.

---

## 🔹 License

MIT License.


