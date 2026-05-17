from typing import Dict, List

from rag.retrieval.config import FusionConfig
from rag.retrieval.scoring import min_max_normalise
from rag.retrieval.types import RankedCandidate, ScoredCandidate


def fuse(
    dense_results: List[ScoredCandidate],
    sparse_results: List[ScoredCandidate],
    config: FusionConfig,
    k: int,
) -> List[RankedCandidate]:
    """Fuse dense and sparse candidate lists into a single ranked list.

    weighted_sum: combined = w_dense * norm(dense) + w_sparse * norm(sparse)
    rrf:          Σ_r 1 / (rrf_k + rank_r(d))
    """
    if k <= 0:
        return []

    if config.strategy == "weighted_sum":
        return _weighted_sum_fusion(dense_results, sparse_results, config, k)
    if config.strategy == "rrf":
        return _rrf_fusion(dense_results, sparse_results, config, k)

    raise ValueError(f"Unknown fusion strategy: '{config.strategy}'. Valid: 'weighted_sum', 'rrf'.")


def _weighted_sum_fusion(
    dense_results: List[ScoredCandidate],
    sparse_results: List[ScoredCandidate],
    config: FusionConfig,
    k: int,
) -> List[RankedCandidate]:
    dense_map: Dict[str, float] = {r["id"]: r["score"] for r in dense_results}
    sparse_map: Dict[str, float] = {r["id"]: r["score"] for r in sparse_results}

    candidate_ids = _combine_ids(dense_map, sparse_map, config.combination)
    if not candidate_ids:
        return []

    if config.normalize_scores:
        dense_norm = _normalise_map(dense_map)
        sparse_norm = _normalise_map(sparse_map)
    else:
        dense_norm = dense_map
        sparse_norm = sparse_map

    results: List[RankedCandidate] = []
    for doc_id in candidate_ids:
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        combined = config.dense_weight * d + config.sparse_weight * s
        results.append({
            "id": doc_id,
            "score": combined,
            "dense_score": dense_map.get(doc_id),
            "sparse_score": sparse_map.get(doc_id),
        })

    results.sort(key=lambda r: (-r["score"], r["id"]))
    return results[:k]


def _normalise_map(score_map: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a {id: score} dict. Equal scores → all 1.0."""
    if not score_map:
        return {}
    ids = list(score_map.keys())
    normalised = min_max_normalise([score_map[i] for i in ids])
    return {doc_id: norm for doc_id, norm in zip(ids, normalised)}


def _rrf_fusion(
    dense_results: List[ScoredCandidate],
    sparse_results: List[ScoredCandidate],
    config: FusionConfig,
    k: int,
) -> List[RankedCandidate]:
    """Reciprocal Rank Fusion. Absent documents get rank = |list| + 1."""
    rrf_k = config.rrf_k

    dense_rank: Dict[str, int] = {r["id"]: i + 1 for i, r in enumerate(dense_results)}
    sparse_rank: Dict[str, int] = {r["id"]: i + 1 for i, r in enumerate(sparse_results)}
    dense_miss = len(dense_results) + 1
    sparse_miss = len(sparse_results) + 1

    dense_map: Dict[str, float] = {r["id"]: r["score"] for r in dense_results}
    sparse_map: Dict[str, float] = {r["id"]: r["score"] for r in sparse_results}

    candidate_ids = _combine_ids(dense_map, sparse_map, config.combination)
    if not candidate_ids:
        return []

    results: List[RankedCandidate] = []
    for doc_id in candidate_ids:
        rrf_score = (
            1.0 / (rrf_k + dense_rank.get(doc_id, dense_miss))
            + 1.0 / (rrf_k + sparse_rank.get(doc_id, sparse_miss))
        )
        results.append({
            "id": doc_id,
            "score": rrf_score,
            "dense_score": dense_map.get(doc_id),
            "sparse_score": sparse_map.get(doc_id),
        })

    results.sort(key=lambda r: (-r["score"], r["id"]))
    return results[:k]


def _combine_ids(
    dense_map: Dict[str, float],
    sparse_map: Dict[str, float],
    combination: str,
) -> List[str]:
    dense_ids = set(dense_map.keys())
    sparse_ids = set(sparse_map.keys())

    if combination == "union":
        return sorted(dense_ids | sparse_ids)
    if combination == "intersection":
        return sorted(dense_ids & sparse_ids)

    raise ValueError(f"Unknown combination mode: '{combination}'. Valid: 'union', 'intersection'.")
