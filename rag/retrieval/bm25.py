from typing import List

from rag.indexing.base import BaseSparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.indexing.sparse.view import InvertedIndexView
from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import BM25Config
from rag.retrieval.scoring import bm25_score_document
from rag.retrieval.types import ScoredCandidate


class BM25Retriever(BaseRetriever):
    """BM25 ranking over the sparse index.

    Query: List[str] (pre-tokenized terms) or str.

        IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        TF_sat(t) = (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        score     = Σ_t IDF(t) * TF_sat(t)
    """

    def __init__(
        self,
        sparse_index: BaseSparseIndex,
        view: InvertedIndexView,
        tokenizer: Tokenizer,
        config: BM25Config,
    ) -> None:
        self._sparse_index = sparse_index
        self._view = view
        self._tokenizer = tokenizer
        self._config = config

    def retrieve_candidates(self, query: object, k: int) -> List[ScoredCandidate]:
        if k <= 0:
            return []

        if isinstance(query, str):
            query_tokens: List[str] = self._tokenizer.tokenize(query)
        elif isinstance(query, list):
            query_tokens = [str(t) for t in query]
        else:
            return []

        if not query_tokens:
            return []

        fetch_k = max(k, self._config.candidate_k)
        sparse_candidates = self._sparse_index.query(query=query_tokens, k=fetch_k)

        if not sparse_candidates:
            return []

        stats = self._view.get_stats()
        n_docs = stats["doc_count"]
        avg_doc_length = stats["avg_doc_length"]
        doc_lengths = stats["doc_lengths"]
        document_frequency = stats["document_frequency"]

        scored: List[ScoredCandidate] = []
        for candidate in sparse_candidates:
            doc_id = candidate["id"]
            score = bm25_score_document(
                query_tokens=query_tokens,
                doc_id=doc_id,
                doc_length=doc_lengths.get(doc_id, 0),
                avg_doc_length=avg_doc_length,
                n_docs=n_docs,
                document_frequency=document_frequency,
                get_tf_fn=self._view.get_tf,
                k1=self._config.k1,
                b=self._config.b,
            )
            scored.append({"id": doc_id, "score": score})

        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:k]
