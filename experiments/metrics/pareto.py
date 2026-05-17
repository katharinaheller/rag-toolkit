"""Pareto-frontier utilities for latency–quality trade-off analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    x: float  # cost axis (latency, smaller is better)
    y: float  # quality axis (recall@k, larger is better)
    extras: dict


def compute_pareto_front(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """Return points that are not strictly dominated.

    ``x`` is minimised, ``y`` is maximised. A point is dominated when another
    has lower (or equal) x and higher (or equal) y, with at least one strict.
    """
    front: List[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q.x <= p.x and q.y >= p.y and (q.x < p.x or q.y > p.y):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p.x)
    return front
