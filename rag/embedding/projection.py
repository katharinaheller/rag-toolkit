from typing import List, Optional


def project_vector(vector: List[float], target_dim: int, method: str) -> List[float]:
    """Project a single vector to target_dim.

    Methods: 'truncate' (lossy prefix), 'mrl' (MRL-trained models only), 'pad' (zero-extend).
    """
    current_dim = len(vector)

    if target_dim is None or current_dim == target_dim:
        return vector

    if target_dim > current_dim:
        if method == "pad":
            return vector + [0.0] * (target_dim - current_dim)
        raise ValueError(
            f"target_dim ({target_dim}) exceeds vector dimension ({current_dim}). "
            "Use method='pad' to extend."
        )

    if method in ("mrl", "truncate", "pad"):
        return vector[:target_dim]

    raise ValueError(f"Unknown projection method: '{method}'. Valid: 'truncate', 'mrl', 'pad'.")


def project_batch(
    vectors: List[List[float]],
    target_dim: int,
    method: str,
) -> List[List[float]]:
    """Project a batch of vectors to target_dim."""
    return [project_vector(v, target_dim, method) for v in vectors]
