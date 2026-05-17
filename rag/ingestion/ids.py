"""Deterministic, collision-resistant ID derivation.

document_id: SHA-256 over a natural key or content hash. Format: "doc_<16 hex>".

Chunk ID strategies:
    positional_chunk_id — SHA-256(document_id:strategy:index). Stable across runs.
    content_chunk_id    — SHA-256(document_id:strategy:index:text). Changes with content.

`strategy` is part of the identity space to prevent cross-strategy collisions.
"""

import hashlib
from typing import Protocol, runtime_checkable


def _sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@runtime_checkable
class ChunkIdFn(Protocol):
    """Callable contract for chunk ID derivation."""

    def __call__(self, document_id: str, index: int, text: str, strategy: str) -> str:
        ...


def positional_chunk_id(document_id: str, index: int, text: str, strategy: str) -> str:
    """Position-keyed ID including strategy namespace. Stable under content changes."""
    return f"chunk_{_sha256_prefix(f'{document_id}:{strategy}:{index}', 16)}"


def content_chunk_id(document_id: str, index: int, text: str, strategy: str) -> str:
    """Content-keyed ID. Changes when text changes — enables drift detection."""
    return f"chunk_{_sha256_prefix(f'{document_id}:{strategy}:{index}:{text}', 16)}"


def document_id_from_text(text: str) -> str:
    """Content-addressed document ID."""
    return f"doc_{_sha256_prefix(text, 16)}"


def document_id_from_natural_key(natural_key: str) -> str:
    """Stable document ID from a caller-supplied primary key."""
    return f"doc_{_sha256_prefix(natural_key.strip(), 16)}"


def resolve_document_id(raw_id: str | None, cleaned_text: str) -> str:
    """Use natural key when non-empty; fall back to content hash."""
    if raw_id is not None and raw_id.strip():
        return document_id_from_natural_key(raw_id)
    return document_id_from_text(cleaned_text)
