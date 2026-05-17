"""Factories for producing canonical RAG data structures used in tests.

Every factory returns plain dicts compatible with the TypedDicts defined in
the production code. Defaults are deterministic so identical calls yield
identical output across runs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def make_chunk(
    chunk_id: str = "chunk_001",
    document_id: str = "doc_001",
    text: str = "Sample chunk text for tests.",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical chunk record."""
    return {
        "id": chunk_id,
        "document_id": document_id,
        "text": text,
        "metadata": dict(metadata) if metadata is not None else {"source": "synthetic"},
    }


def make_chunks(n: int, document_id: str = "doc_001", prefix: str = "chunk") -> List[Dict[str, Any]]:
    """Build n distinct chunks belonging to one document."""
    return [
        make_chunk(
            chunk_id=f"{prefix}_{i:04d}",
            document_id=document_id,
            text=f"This is chunk number {i} with stable deterministic content.",
            metadata={"source": "synthetic", "index": i},
        )
        for i in range(n)
    ]


def make_dense_vector(dim: int = 8, seed: int = 0) -> List[float]:
    """Build a deterministic dense vector of fixed dimension."""
    digest = hashlib.sha256(f"vec-{seed}".encode("utf-8")).digest()
    raw = [(b - 128) / 128.0 for b in digest[:dim]]
    if len(raw) < dim:
        raw = raw + [0.0] * (dim - len(raw))
    return raw


def make_embedding_vector(
    chunk_id: str = "chunk_001",
    document_id: str = "doc_001",
    dim: int = 8,
    seed: int = 0,
    sparse: Optional[Dict[str, float]] = None,
    embedding_type: str = "dense",
) -> Dict[str, Any]:
    """Build a canonical EmbeddingVector dict."""
    return {
        "id": f"emb_{chunk_id}",
        "chunk_id": chunk_id,
        "document_id": document_id,
        "embedding": make_dense_vector(dim=dim, seed=seed) if embedding_type != "sparse" else None,
        "sparse_embedding": sparse,
        "embedding_type": embedding_type,
        "model_type": "test-model",
        "projection_method": "none",
        "metadata": {"source": "synthetic"},
    }


def make_stored_document(
    doc_id: str = "emb_chunk_001",
    chunk_id: str = "chunk_001",
    document_id: str = "doc_001",
    text: str = "Stored document text.",
    dense: Optional[List[float]] = None,
    sparse: Optional[Dict[str, float]] = None,
    embedding_type: str = "dense",
) -> Dict[str, Any]:
    """Build a canonical StoredDocument dict for indexing tests."""
    return {
        "id": doc_id,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "dense_vector": dense if dense is not None else make_dense_vector(),
        "sparse_vector": sparse,
        "embedding_type": embedding_type,
        "metadata": {"source": "synthetic"},
    }


def make_retrieval_result(
    rid: str = "doc_001",
    chunk_id: str = "chunk_001",
    document_id: str = "doc_001",
    score: float = 0.9,
    text: str = "Sample retrieval text.",
) -> Dict[str, Any]:
    """Build a canonical RetrievalResult dict."""
    return {
        "id": rid,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "score": score,
        "retrieval_score": score,
        "text": text,
        "metadata": {"source": "synthetic"},
    }
