from typing import Dict, List, Set

from rag.indexing.sparse.inverted_index import InvertedIndex
from rag.indexing.types import InvertedIndexStats


class InvertedIndexView:
    """Read-only interface to InvertedIndex for the retrieval module.

    This is the only access path from outside inverted_index.py. Direct
    attribute access on InvertedIndex is forbidden.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

    def get_tf(self, token: str, doc_id: str) -> int:
        """Term frequency in O(1). Returns 0 if absent."""
        return self._index._get_tf(token, doc_id)

    def get_stats(self) -> InvertedIndexStats:
        """Corpus statistics for BM25/TF-IDF scoring."""
        return self._index._get_stats()

    def get_matching_doc_ids(self, tokens: List[str]) -> Set[str]:
        """Doc IDs containing at least one of the given tokens."""
        return self._index._get_matching_doc_ids(tokens)

    def get_vocabulary(self) -> Dict[str, int]:
        """token -> integer ID vocabulary mapping."""
        return self._index._get_vocabulary()
