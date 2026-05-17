from typing import List

from rag.indexing.base import BaseSparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.indexing.sparse.view import InvertedIndexView
from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import TFIDFConfig
from rag.retrieval.scoring import tfidf_score_document
from rag.retrieval.types import ScoredCandidate


class TFIDFRetriever(BaseRetriever):
    """TF-IDF retrieval. No TF saturation or length normalization (use BM25 for those)."""

    def __init__(
        self,
        sparse_index: BaseSparseIndex,
        view: InvertedIndexView,
        tokenizer: Tokenizer,
        config: TFIDFConfig,
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
        document_frequency = stats["document_frequency"]

        scored: List[ScoredCandidate] = []
        for candidate in sparse_candidates:
            score = tfidf_score_document(
                query_tokens=query_tokens,
                doc_id=candidate["id"],
                n_docs=n_docs,
                document_frequency=document_frequency,
                get_tf_fn=self._view.get_tf,
                sublinear_tf=self._config.sublinear_tf,
            )
            scored.append({"id": candidate["id"], "score": score})

        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:k]
