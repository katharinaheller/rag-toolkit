import hashlib
from typing import Optional


def embedding_id(
    chunk_id: str,
    model_name: str,
    model_version: Optional[str] = None,
) -> str:
    """Deterministic SHA-256 ID for an embedding.

    Including model_version ensures IDs change when model weights change,
    preventing silent retrieval mismatches across experiments.
    """
    version_tag = model_version if model_version is not None else "unversioned"
    raw = f"{chunk_id}|{model_name}|{version_tag}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
