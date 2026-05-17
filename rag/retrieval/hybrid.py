from typing import Dict, List, Tuple

from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import FusionConfig
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fusion import fuse
from rag.retrieval.types import HybridQuery, RankedCandidate, ScoredCandidate


class HybridRetriever(BaseRetriever):
    """Combine dense and sparse retrieval via score fusion.

    Query: HybridQuery dict with optional "dense" and "sparse" keys.
    Each leg uses its own candidate_k.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BaseRetriever,
        fusion_config: FusionConfig,
        dense_candidate_k: int,
        sparse_candidate_k: int,
    ) -> None:
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._fusion_config = fusion_config
        self._dense_candidate_k = dense_candidate_k
        self._sparse_candidate_k = sparse_candidate_k

    def retrieve_candidates(self, query: object, k: int) -> List[ScoredCandidate]:
        if k <= 0:
            return []
        _, _, fused = self._run(query, k)
        return [{"id": c["id"], "score": c["score"]} for c in fused]

    def retrieve_with_details(
        self, query: object, k: int
    ) -> Tuple[List[ScoredCandidate], List[ScoredCandidate], List[RankedCandidate]]:
        """Return dense, sparse, and fused candidates (for tracing)."""
        return self._run(query, k)

    def _run(
        self, query: object, k: int
    ) -> Tuple[List[ScoredCandidate], List[ScoredCandidate], List[RankedCandidate]]:
        if not isinstance(query, dict):
            return [], [], []

        hybrid_query: HybridQuery = query  # type: ignore

        dense_q = hybrid_query.get("dense")
        sparse_q = hybrid_query.get("sparse")

        dense_candidates: List[ScoredCandidate] = []
        if isinstance(dense_q, list) and dense_q:
            dense_candidates = self._dense_retriever.retrieve_candidates(
                query=dense_q, k=self._dense_candidate_k
            )

        sparse_candidates: List[ScoredCandidate] = []
        if sparse_q is not None:
            valid_sparse = (isinstance(sparse_q, list) and sparse_q) or (
                isinstance(sparse_q, dict) and sparse_q
            )
            if valid_sparse:
                sparse_candidates = self._sparse_retriever.retrieve_candidates(
                    query=sparse_q, k=self._sparse_candidate_k
                )

        fused = fuse(dense_candidates, sparse_candidates, self._fusion_config, k=k)
        return dense_candidates, sparse_candidates, fused
