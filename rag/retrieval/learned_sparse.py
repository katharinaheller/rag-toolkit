"""Learned sparse dot-product retriever.

Scoring is the dot product over continuous sparse weight vectors (token_id →
weight) produced by models like BGE-M3 (lexical_weights). Not BM25.

Two execution paths:
    use_index_pruning=True  — builds an in-memory inverted index at construction.
                              Query cost: O(Σ |posting_list(t)| for t ∈ query).
    use_index_pruning=False — full sequential scan. Use when RAM is constrained
                              or documents change frequently.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set

from rag.indexing.store import DocumentStore
from rag.indexing.types import StoredDocument
from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import LearnedSparseConfig
from rag.retrieval.scoring import sparse_dot_product
from rag.retrieval.types import ScoredCandidate


class LearnedSparseRetriever(BaseRetriever):
    """Learned sparse dot-product retriever."""

    def __init__(self, document_store: DocumentStore, config: LearnedSparseConfig) -> None:
        self._document_store = document_store
        self._config = config
        self._token_to_doc_ids: Optional[Dict[str, Set[str]]] = None
        self._doc_index: Optional[Dict[str, StoredDocument]] = None

        if config.use_index_pruning:
            self._build_pruning_index()

    def _build_pruning_index(self) -> None:
        token_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        doc_index: Dict[str, StoredDocument] = {}

        for doc in self._document_store.stream_all():
            sparse_vec = doc.get("sparse_vector")
            if not sparse_vec:
                continue
            doc_id = doc["id"]
            doc_index[doc_id] = doc
            for token_id in sparse_vec:
                token_to_doc_ids[token_id].add(doc_id)

        self._token_to_doc_ids = dict(token_to_doc_ids)
        self._doc_index = doc_index

    def retrieve_candidates(self, query: object, k: int) -> List[ScoredCandidate]:
        if k <= 0:
            return []
        if not isinstance(query, dict) or not query:
            return []

        query_sparse: Dict[str, float] = query

        if self._token_to_doc_ids is not None and self._doc_index is not None:
            scored = self._score_pruned(query_sparse)
        else:
            scored = self._score_full_scan(query_sparse)

        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:k]

    def _score_pruned(self, query_sparse: Dict[str, float]) -> List[ScoredCandidate]:
        candidate_ids: Set[str] = set()
        for token_id in query_sparse:
            posting = self._token_to_doc_ids.get(token_id)  # type: ignore[union-attr]
            if posting:
                candidate_ids.update(posting)

        scored: List[ScoredCandidate] = []
        for doc_id in candidate_ids:
            doc = self._doc_index.get(doc_id)  # type: ignore[union-attr]
            if doc is None:
                continue
            doc_sparse = doc.get("sparse_vector")
            if not doc_sparse:
                continue
            score = sparse_dot_product(query_sparse, doc_sparse)
            if score <= 0.0 or score < self._config.min_score:
                continue
            scored.append({"id": doc_id, "score": score})

        return scored

    def _score_full_scan(self, query_sparse: Dict[str, float]) -> List[ScoredCandidate]:
        scored: List[ScoredCandidate] = []
        for doc in self._document_store.stream_all():
            doc_sparse = doc.get("sparse_vector")
            if not doc_sparse:
                continue
            score = sparse_dot_product(query_sparse, doc_sparse)
            if score <= 0.0 or score < self._config.min_score:
                continue
            scored.append({"id": doc["id"], "score": score})
        return scored
