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

In this section, we briefly introduce the evaluation of Information Retrieval Systems (IRS) [bida2025ri, manning2008evaluation, verberne2024evaluation].

IRS can be divided into two main categories based on their output: (1) unranked retrieval systems, which provide binary or multi-class relevance judgments; and (2) ranked retrieval systems, which return documents ordered by relevance.

Evaluation metrics are not always suitable for both types of systems. Some metrics rely on discrete relevance classes to be computed, such as precision, recall, and F1-score. Since our models belong to the second category, these metrics are not well adapted to our task. We will discuss alternative metrics and explain the limitations of class-based metrics in this context in the following sections.

### Set-Based Evaluation Metrics

**Precision (P)** is the fraction of retrieved documents that are relevant.

$$
\text{Precision} = \frac{|\text{relevant items retrieved}|}{|\text{retrieved items}|}
$$

**Recall (R)** is the fraction of relevant documents that are retrieved.

$$
\text{Recall} = \frac{|\text{relevant items retrieved}|}{|\text{relevant items}|}
$$

**F1-score** is the weighted harmonic mean of precision and recall.

$$
\text{F1} = \frac{2PR}{P+R}
$$

***Observation:***
As discussed above, precision, recall, and F1-score are *set-based* evaluation measures, meaning they are computed over unordered sets of retrieved documents and assume a binary notion of relevance.

When the entire ranked list is considered as the retrieved set, recall is always equal to one. Indeed, recall measures the proportion of relevant documents retrieved, and since a ranked retrieval system eventually returns all documents, all relevant documents are retrieved regardless of their rank. Consequently, recall is independent of the model's ranking quality.

Precision is equally uninformative in this setting. The number of retrieved documents is fixed across models, and the number of relevant documents retrieved corresponds to the total number of relevant documents for a given query. Therefore, precision varies only across queries, depending on the number of relevant documents, but **not across models**.

To address these limitations, ranked retrieval evaluation relies on metrics computed over the top-$k$ retrieved documents (e.g., Precision@K and R-Precision) as well as rank-aware measures that assess the quality of the ranking order, such as precision–recall curves and interpolated precision.

### Rank-Based Evaluation Metrics

#### Precision–Recall Curve

To adapt precision and recall to ranked retrieval, precision@$k$ and recall@$k$ are computed at each rank $k$ where a relevant document is retrieved. Plotting precision as a function of recall over these points yields the precision–recall curve, which characterizes how retrieval effectiveness evolves across the ranking.

The ideal P–R curve starts at precision = 1 and remains close to 1 across all recall levels.

#### Interpolated Precision–Recall Curve

Due to the zigzag nature of precision–recall curves, an interpolated version is often used. The interpolated precision at recall level $r$ is defined as:

$$
p_{\text{interp}}(r) = \max_{r' \geq r} p(r')
$$


#### Average Precision (AP)

Average Precision summarizes the precision–recall curve into a single value by averaging the precision obtained at the rank positions where relevant documents are retrieved:

$$
AP = \frac{1}{R} \sum_{k=1}^{N} P(k) \cdot I(k)
$$

where $R$ is the total number of relevant documents and $I(k)$ is an indicator function equal to 1 if the document at rank $k$ is relevant. AP rewards systems that retrieve relevant documents early in the ranking.

#### Mean Average Precision (MAP)

MAP is the mean of the Average Precision scores over all queries:

$$
MAP = \frac{1}{Q} \sum_{q=1}^{Q} AP(q)
$$

It provides a **single global measure** of ranking effectiveness and is widely used for comparing retrieval models.

#### Interpolated MAP

Interpolated MAP is computed using interpolated precision values instead of raw precision, ensuring smoother and more robust comparisons between models, especially when relevance judgments are sparse.

#### Precision@K (P@K)

Precision@K measures the proportion of relevant documents among the top-$K$ retrieved results:

$$
P@K = \frac{|\text{ relevant documents in top-}K|}{K}
$$

This metric focuses on early precision and reflects user satisfaction when **only the first few results are examined**. In this work, we report $P@5$ and $P@10$.

#### R-Precision

R-Precision computes precision at rank $R$, where $R$ is the number of relevant documents for a given query:

$$
R\text{-Precision} = P@R
$$

This metric provides a balanced evaluation point that adapts to the difficulty of each query.

#### Reciprocal Rank (RR)

Reciprocal Rank measures how early the first relevant document is retrieved:

$$
RR = \frac{1}{\text{rank of first relevant document}}
$$

It strongly emphasizes early retrieval and is particularly suited to scenarios where **finding at least one relevant document quickly is critical**.

#### Discounted Cumulative Gain (DCG)

To understand DCG, we first consider Cumulative Gain (CG), which is defined as the sum of the relevance judgments of the retrieved documents:

$$
\text{CG@}n = \sum_{i=1}^{n} r_i
$$

where:

- $n$ is the number of results in the ranked list that are considered,
- $r_i$ is the relevance judgment of the document at rank $i$.

CG assumes that documents with higher relevance judgments contribute more to the overall usefulness of the ranked list, but it does not take the position of a document into account.

To model the fact that users are less likely to examine documents at lower ranks, Discounted Cumulative Gain (DCG) introduces a logarithmic discount factor:

$$
\text{DCG@}n = r_1 + \sum_{i=2}^{n} \frac{r_i}{\log_2 i}
$$

This formulation discounts the gain of documents appearing at lower ranks, reflecting reduced user attention.

#### Normalized Discounted Cumulative Gain (nDCG)

The magnitude of DCG depends on the number and distribution of relevant documents for a given query, making scores not directly comparable across queries. To address this, DCG is normalized by the ideal DCG (iDCG), which corresponds to the DCG of a perfectly ranked list:

$$
\text{nDCG@}n = \frac{\text{DCG@}n}{\text{iDCG@}n}
$$

where iDCG@$n$ is obtained by sorting documents in decreasing order of their relevance judgments (i.e., all highly relevant documents first, followed by relevant and non-relevant documents).

As a result, nDCG values lie in the range $[0,1]$, enabling meaningful comparison of retrieval performance across different queries.

***Observation:***
Relevance assessments in the MEDLINE dataset are binary, i.e., $r_i \in \{0,1\}$, indicating whether a document is relevant or non-relevant. Consequently, DCG cannot exploit graded relevance levels, since all relevant documents contribute the same gain. Nevertheless, DCG remains informative because it still rewards rankings that place relevant documents at higher positions through its logarithmic discount factor. In this setting, DCG and nDCG act as rank-sensitive measures that emphasize early retrieval rather than relevance intensity.

---

## 📌 Notes

* MEDLINE queries are included in the dataset (`MED.QRY`) and preprocessed before retrieval.
* Relevance judgments are binary.
* Modular design allows adding new models and metrics easily.

---

## 🔹 License

MIT License.


