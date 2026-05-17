"""Stateless scoring functions for all retrieval strategies.

min_max_normalise returns all 1.0 when max == min, preserving the retrieval
leg's contribution in weighted fusion.
"""

import math
from typing import Callable, Dict, List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity. Returns 0.0 for zero-norm vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def bm25_idf(df: int, n: int) -> float:
    """BM25 IDF (Robertson-Sparck Jones smoothed).

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    """
    if n <= 0 or df <= 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def bm25_tf_component(tf: int, doc_length: int, avg_doc_length: float, k1: float, b: float) -> float:
    """BM25 TF saturation + length normalization component."""
    if tf == 0 or avg_doc_length == 0.0:
        return 0.0
    length_norm = 1.0 - b + b * (doc_length / avg_doc_length)
    return (tf * (k1 + 1.0)) / (tf + k1 * length_norm)


def bm25_score_document(
    query_tokens: List[str],
    doc_id: str,
    doc_length: int,
    avg_doc_length: float,
    n_docs: int,
    document_frequency: Dict[str, int],
    get_tf_fn: Callable[[str, str], int],
    k1: float,
    b: float,
) -> float:
    """Full BM25 score for one (query, document) pair. Unique query tokens only."""
    score = 0.0
    for token in set(query_tokens):
        tf = get_tf_fn(token, doc_id)
        if tf == 0:
            continue
        df = document_frequency.get(token, 0)
        idf = bm25_idf(df=df, n=n_docs)
        tf_comp = bm25_tf_component(
            tf=tf, doc_length=doc_length, avg_doc_length=avg_doc_length, k1=k1, b=b
        )
        score += idf * tf_comp
    return score


def tfidf_tf(raw_tf: int, sublinear: bool) -> float:
    """TF component for TF-IDF. Sublinear form: 1 + log(raw_tf)."""
    if raw_tf <= 0:
        return 0.0
    if sublinear:
        return 1.0 + math.log(raw_tf)
    return float(raw_tf)


def tfidf_idf(df: int, n: int) -> float:
    """Smoothed IDF for TF-IDF: log((N + 1) / (df + 1)) + 1."""
    if n <= 0 or df <= 0:
        return 1.0
    return math.log((n + 1.0) / (df + 1.0)) + 1.0


def tfidf_score_document(
    query_tokens: List[str],
    doc_id: str,
    n_docs: int,
    document_frequency: Dict[str, int],
    get_tf_fn: Callable[[str, str], int],
    sublinear_tf: bool,
) -> float:
    """Full TF-IDF score for one (query, document) pair."""
    score = 0.0
    for token in set(query_tokens):
        raw_tf = get_tf_fn(token, doc_id)
        if raw_tf == 0:
            continue
        score += tfidf_tf(raw_tf, sublinear=sublinear_tf) * tfidf_idf(
            df=document_frequency.get(token, 0), n=n_docs
        )
    return score


def sparse_dot_product(query_sparse: Dict[str, float], doc_sparse: Dict[str, float]) -> float:
    """Dot product between two sparse vectors. Iterates over the shorter dict."""
    if not query_sparse or not doc_sparse:
        return 0.0
    if len(query_sparse) > len(doc_sparse):
        query_sparse, doc_sparse = doc_sparse, query_sparse
    score = 0.0
    for token, q_weight in query_sparse.items():
        d_weight = doc_sparse.get(token, 0.0)
        if d_weight != 0.0:
            score += q_weight * d_weight
    return score


def min_max_normalise(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]. Equal scores → all 1.0."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    span = max_s - min_s
    return [(s - min_s) / span for s in scores]
