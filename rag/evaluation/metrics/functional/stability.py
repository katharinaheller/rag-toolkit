"""Stability metrics and bootstrap confidence intervals.

Stability quantifies how much a retriever's output changes across reseeded
re-runs (or across small input perturbations). A retriever with high recall
but low stability is harder to trust in production.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple


def std_dev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return std_dev(values) / m if m else 0.0


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` for a confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) < 2:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    overall_mean = sum(values) / len(values)
    return overall_mean, means[lo_idx], means[hi_idx]


def rank_stability(
    rank_lists_per_run: List[List[str]],
    k: int,
) -> float:
    """Mean pairwise Jaccard@k between rank lists across repeated runs."""
    if len(rank_lists_per_run) < 2:
        return 1.0
    sets = [set(rl[:k]) for rl in rank_lists_per_run]
    scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            score = (len(sets[i] & sets[j]) / len(union)) if union else 0.0
            scores.append(score)
    return sum(scores) / len(scores) if scores else 1.0
