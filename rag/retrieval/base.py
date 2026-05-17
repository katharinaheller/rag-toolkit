from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from rag.retrieval.types import ScoredCandidate


class BaseRetriever(ABC):
    """Abstract base for all retrieval strategies.

    Invariants:
        - Returns only (id, score). Never text or metadata.
        - Results sorted by score DESC, id ASC (deterministic).
        - Invalid/empty query, k <= 0, or no candidates → returns [].

    Query types by implementation:
        DenseRetriever:         List[float]
        BM25/TFIDFRetriever:    List[str] or str
        LearnedSparseRetriever: Dict[str, float]
        HybridRetriever:        dict with "dense"/"sparse" keys
    """

    @abstractmethod
    def retrieve_candidates(self, query: Any, k: int) -> List[ScoredCandidate]:
        """Return top-k scored candidates for the given query."""
        raise NotImplementedError("Subclasses must implement retrieve_candidates().")
