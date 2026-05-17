from abc import ABC, abstractmethod
from typing import List

from rag.indexing.types import (
    DenseCandidateResult,
    DenseIndexEntry,
    DenseQuery,
    SparseCandidate,
    SparseQuery,
    TokenizedEntry,
)


class BaseDenseIndex(ABC):
    """Contract for dense vector index backends.

    Input: DenseIndexEntry (id + vector). Output: DenseCandidateResult (id + score).
    Never stores or returns text or metadata.
    """

    @abstractmethod
    def add(self, entries: List[DenseIndexEntry]) -> None:
        """Ingest dense entries. Incremental: may be called multiple times."""

    @abstractmethod
    def query(self, query: DenseQuery, k: int) -> List[DenseCandidateResult]:
        """Return top-k candidates sorted by descending score, then id."""

    @abstractmethod
    def persist(self) -> None:
        """Persist index state to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load index state from disk."""


class BaseSparseIndex(ABC):
    """Contract for sparse lexical index backends.

    Input: TokenizedEntry. Output: SparseCandidate (id only).
    Scoring belongs exclusively to the retrieval module.
    """

    @abstractmethod
    def add(self, entries: List[TokenizedEntry]) -> None:
        """Ingest tokenized entries. Incremental: may be called multiple times."""

    @abstractmethod
    def query(self, query: SparseQuery, k: int) -> List[SparseCandidate]:
        """Return matching candidates sorted by id (deterministic, unscored)."""

    @abstractmethod
    def persist(self) -> None:
        """Persist index state to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load index state from disk."""
