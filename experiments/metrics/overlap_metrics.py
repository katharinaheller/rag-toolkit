"""Retrieval-overlap and rank-correlation metrics.

These metrics quantify how much two retrievers agree on which chunks belong
in the top-k for a query. Low overlap means complementary evidence — the
empirical justification for hybrid retrieval.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap_at_k(ids_a: Sequence[str], ids_b: Sequence[str], k: int) -> float:
    return jaccard(set(ids_a[:k]), set(ids_b[:k]))


def kendall_tau_topk(ids_a: Sequence[str], ids_b: Sequence[str], k: int) -> float:
    """Kendall's tau over the union of top-k. Missing ranks get rank = k+1."""
    top_a = list(ids_a[:k])
    top_b = list(ids_b[:k])
    union = list(set(top_a) | set(top_b))
    if len(union) < 2:
        return 0.0

    rank_a = {x: top_a.index(x) if x in top_a else k for x in union}
    rank_b = {x: top_b.index(x) if x in top_b else k for x in union}

    concordant, discordant = 0, 0
    for i in range(len(union)):
        for j in range(i + 1, len(union)):
            xi, xj = union[i], union[j]
            ai = rank_a[xi] - rank_a[xj]
            bi = rank_b[xi] - rank_b[xj]
            if ai == 0 or bi == 0:
                continue
            if (ai > 0) == (bi > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def pairwise_overlap_matrix(
    retrieved_per_retriever: Dict[str, Dict[str, List[str]]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    """Compute mean top-k Jaccard overlap across queries for each retriever pair.

    ``retrieved_per_retriever[retr_key][query_id]`` is a ranked list of ids.
    """
    keys = sorted(retrieved_per_retriever.keys())
    matrix: Dict[str, Dict[str, float]] = {a: {b: 0.0 for b in keys} for a in keys}

    for a in keys:
        for b in keys:
            if a == b:
                matrix[a][b] = 1.0
                continue
            qa = retrieved_per_retriever[a]
            qb = retrieved_per_retriever[b]
            shared = set(qa.keys()) & set(qb.keys())
            if not shared:
                matrix[a][b] = 0.0
                continue
            scores = [overlap_at_k(qa[q], qb[q], k) for q in shared]
            matrix[a][b] = sum(scores) / len(scores)
    return matrix
