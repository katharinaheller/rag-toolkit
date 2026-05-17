from typing import Any, Dict, List, Optional, TypedDict, Union


class HybridQuery(TypedDict, total=False):
    """Structured query for HybridRetriever."""
    dense: List[float]
    sparse: Union[List[str], Dict[str, float]]


class ScoredCandidate(TypedDict):
    """Internal intermediate: id + score, before document store join."""
    id: str
    score: float


class RankedCandidate(TypedDict):
    """Fused candidate from fusion strategies, before document store join."""
    id: str
    score: float
    dense_score: Optional[float]
    sparse_score: Optional[float]


class _RetrievalResultRequired(TypedDict):
    """Required fields guaranteed in every RetrievalResult."""
    id: str
    chunk_id: str
    document_id: str
    score: float
    retrieval_score: float
    text: str
    metadata: Dict[str, Any]


class RetrievalResult(_RetrievalResultRequired, total=False):
    """Final output record returned to the caller."""
    rerank_score: float
    dense_score: float
    sparse_score: float


class RetrievalTrace(TypedDict):
    """Full intermediate data snapshot from one retrieve_with_trace() call."""
    mode: str
    query: str
    query_tokens: Optional[List[str]]
    query_sparse_vector: Optional[Dict[str, float]]
    dense_candidates: List[ScoredCandidate]
    sparse_candidates: List[ScoredCandidate]
    fused_candidates: List[RankedCandidate]
    final_results: List[RetrievalResult]
