from typing import Any, Dict, List, Optional, TypedDict


class EmbeddingVector(TypedDict):
    id: str                                       # SHA-256(chunk_id|model_name|version)
    chunk_id: str
    document_id: str
    embedding: Optional[List[float]]              # dense vector; None for sparse-only
    sparse_embedding: Optional[Dict[str, float]]  # token_id -> weight; None for dense-only
    embedding_type: str                           # "dense" | "sparse" | "hybrid"
    model_type: str
    projection_method: str                        # "none" | "truncate" | "mrl" | "pad"
    metadata: Dict[str, Any]


class EmbeddingResult(TypedDict):
    vectors: List[EmbeddingVector]
