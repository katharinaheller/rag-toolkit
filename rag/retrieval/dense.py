from typing import List

from rag.indexing.base import BaseDenseIndex
from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import DenseRetrievalConfig
from rag.retrieval.types import ScoredCandidate


class DenseRetriever(BaseRetriever):
    """Retrieve candidates via the dense ANN index.

    Query: List[float]. Score semantics depend on the configured metric:
    cosine in [-1, 1], dot unbounded, l2 negative distances.

    Queries with max(k, candidate_k) to provide a sufficient pool for fusion
    or reranking downstream.
    """

    def __init__(self, dense_index: BaseDenseIndex, config: DenseRetrievalConfig) -> None:
        self._index = dense_index
        self._config = config

    def retrieve_candidates(self, query: object, k: int) -> List[ScoredCandidate]:
        if k <= 0:
            return []
        if not isinstance(query, list) or not query:
            return []

        fetch_k = max(k, self._config.candidate_k)
        raw_results = self._index.query(query=query, k=fetch_k)

        candidates: List[ScoredCandidate] = [
            {"id": r["id"], "score": r["score"]} for r in raw_results
        ]
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        return candidates[:k]
