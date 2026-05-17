from typing import Any, Dict, List, Optional, TypedDict

DenseQuery = List[float]
SparseQuery = List[str]


class DenseIndexEntry(TypedDict):
    """Text-free input to a dense index backend."""
    id: str
    dense_vector: List[float]


class TokenizedEntry(TypedDict):
    """Pre-tokenized input to a sparse index backend."""
    id: str
    tokens: List[str]


class StoredDocument(TypedDict):
    """Full document record — single source of truth for text and metadata.

    sparse_vector holds model-learned weights (e.g. BGE-M3 lexical output).
    """
    id: str
    chunk_id: str
    document_id: str
    text: str
    dense_vector: Optional[List[float]]
    sparse_vector: Optional[Dict[str, float]]
    embedding_type: str
    metadata: Dict[str, Any]


class DenseCandidateResult(TypedDict):
    """Output of a dense index query."""
    id: str
    score: float


class SparseCandidate(TypedDict):
    """Output of a sparse index query. Unscored by design."""
    id: str


class InvertedIndexStats(TypedDict):
    """Corpus-level statistics for BM25/TF-IDF scoring."""
    doc_count: int
    avg_doc_length: float
    doc_lengths: Dict[str, int]
    document_frequency: Dict[str, int]
