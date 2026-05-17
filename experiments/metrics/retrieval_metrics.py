"""Retrieval IR metrics computed against provenance-anchored gold.

Each metric takes a ranked list of retrieved chunk ids (or document ids) and
a gold set, and returns a scalar in [0, 1]. We work with chunk ids when the
gold labels are chunk-level (the default for queries derived from chunks).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Set


def hit_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if k <= 0 or not gold_ids:
        return 0.0
    return 1.0 if any(i in gold_ids for i in retrieved_ids[:k]) else 0.0


def precision_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for i in top if i in gold_ids) / float(len(top))


def recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = retrieved_ids[:k]
    found = sum(1 for i in top if i in gold_ids)
    return found / float(len(gold_ids))


def reciprocal_rank(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    if not gold_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids or k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, doc_id in enumerate(top) if doc_id in gold_ids
    )
    ideal_n = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def aggregate_retrieval_metrics(
    rows: List[dict],
    gold_lookup: dict,
    k_values: List[int],
    id_field: str = "chunk_id",
) -> dict:
    """Aggregate per-query metrics from a per-rank retrieval table.

    ``rows`` must be a list of dicts with ``query_id``, ``rank``, ``chunk_id``,
    ``document_id``. ``gold_lookup`` maps ``query_id`` → set of gold ids.
    Returns a dict keyed by metric name (e.g. ``"recall@5"``) → mean value.
    """
    # Group retrieved ids per query, sorted by rank.
    by_query: dict = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    for qid in by_query:
        by_query[qid].sort(key=lambda r: r["rank"])

    out: dict = {}
    for k in k_values:
        precisions, recalls, hits, ndcgs = [], [], [], []
        rrs = []
        for qid, group in by_query.items():
            gold = gold_lookup.get(qid, set())
            ids = [r[id_field] for r in group]
            precisions.append(precision_at_k(ids, gold, k))
            recalls.append(recall_at_k(ids, gold, k))
            hits.append(hit_at_k(ids, gold, k))
            ndcgs.append(ndcg_at_k(ids, gold, k))
            rrs.append(reciprocal_rank(ids[:k], gold))
        out[f"precision@{k}"] = mean(precisions)
        out[f"recall@{k}"] = mean(recalls)
        out[f"hit@{k}"] = mean(hits)
        out[f"ndcg@{k}"] = mean(ndcgs)
        out[f"mrr@{k}"] = mean(rrs)
    return out
