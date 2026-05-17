import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: List[float], b: List[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def l2_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def l2_normalize(vector: List[float]) -> List[float]:
    """L2-normalize a vector. Returns the original if norm is zero.

    Used by FAISSIndex: inner product on unit vectors equals cosine similarity,
    so IndexFlatIP serves as a correct cosine index.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def validate_real_vector(vector: List[float], field_name: str) -> None:
    """Raise TypeError/ValueError if vector contains non-finite or non-real values."""
    from numbers import Real
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must contain only real numbers.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must contain only finite real numbers.")
