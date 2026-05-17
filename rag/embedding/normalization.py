import math
from typing import List


def l2_normalize(vector: List[float]) -> List[float]:
    """L2-normalize a single vector. Returns the original if norm is zero."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def normalize_batch(vectors: List[List[float]]) -> List[List[float]]:
    """L2-normalize a batch of vectors."""
    return [l2_normalize(v) for v in vectors]
