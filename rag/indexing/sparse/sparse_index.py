from pathlib import Path
from typing import List

from rag.indexing.base import BaseSparseIndex
from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.inverted_index import InvertedIndex
from rag.indexing.sparse.view import InvertedIndexView
from rag.indexing.types import SparseCandidate, SparseQuery, TokenizedEntry


class SparseIndex(BaseSparseIndex):
    """Sparse lexical index backed by InvertedIndex."""

    def __init__(self, config: SparseIndexConfig, index_dir: Path) -> None:
        self._config = config
        self._index_dir = index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._inverted_index = InvertedIndex()
        self._view = InvertedIndexView(self._inverted_index)

    def add(self, entries: List[TokenizedEntry]) -> None:
        self._inverted_index._build(entries)

    def query(self, query: SparseQuery, k: int) -> List[SparseCandidate]:
        if k <= 0:
            return []
        if not isinstance(query, list):
            raise TypeError(f"SparseIndex.query expects List[str], got {type(query).__name__}.")
        if not query:
            return []
        if any(not isinstance(t, str) for t in query):
            raise TypeError("SparseIndex.query expects List[str], got list with non-string values.")

        candidate_ids = self._view.get_matching_doc_ids(query)
        return sorted([{"id": doc_id} for doc_id in candidate_ids], key=lambda r: r["id"])[:k]

    def get_view(self) -> InvertedIndexView:
        return self._view

    def persist(self) -> None:
        self._inverted_index._persist(self._index_dir)

    def load(self) -> None:
        self._inverted_index._load(self._index_dir)
