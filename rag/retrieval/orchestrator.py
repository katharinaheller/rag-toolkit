import logging
from typing import Any, Dict, List, Optional, Tuple

from rag.embedding.base import BaseEmbedder
from rag.indexing.base import BaseDenseIndex, BaseSparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.indexing.sparse.view import InvertedIndexView
from rag.indexing.store import DocumentStore
from rag.indexing.types import StoredDocument

from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.config import RetrievalConfig
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.learned_sparse import LearnedSparseRetriever
from rag.retrieval.reranker import BaselineReranker
from rag.retrieval.tfidf import TFIDFRetriever
from rag.retrieval.types import (
    HybridQuery,
    RankedCandidate,
    RetrievalResult,
    RetrievalTrace,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_DENSE_MODES = frozenset({"dense", "hybrid"})
_LEARNED_SPARSE_MODES = frozenset({"sparse_learned"})
_LEXICAL_MODES = frozenset({"sparse_bm25", "sparse_tfidf"})


class RetrievalOrchestrator:
    """End-to-end retrieval pipeline: query string → List[RetrievalResult]."""

    def __init__(
        self,
        document_store: DocumentStore,
        config: RetrievalConfig,
        embedder: Optional[BaseEmbedder] = None,
        dense_index: Optional[BaseDenseIndex] = None,
        sparse_index: Optional[BaseSparseIndex] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._config = config
        self._document_store = document_store
        self._embedder = embedder
        self._tokenizer = tokenizer

        _validate_dependencies(config, embedder, dense_index, sparse_index, tokenizer)

        logger.debug("RetrievalOrchestrator: loading document store.")
        self._doc_by_id: Dict[str, StoredDocument] = document_store.load_by_id()
        logger.debug("RetrievalOrchestrator: document store ready. count=%d", len(self._doc_by_id))

        self._dense_retriever: Optional[DenseRetriever] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._tfidf_retriever: Optional[TFIDFRetriever] = None
        self._learned_retriever: Optional[LearnedSparseRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None

        if dense_index is not None:
            self._dense_retriever = DenseRetriever(dense_index=dense_index, config=config.dense)

        if sparse_index is not None and tokenizer is not None:
            view: InvertedIndexView = sparse_index.get_view()
            self._bm25_retriever = BM25Retriever(
                sparse_index=sparse_index, view=view, tokenizer=tokenizer, config=config.bm25
            )
            self._tfidf_retriever = TFIDFRetriever(
                sparse_index=sparse_index, view=view, tokenizer=tokenizer, config=config.tfidf
            )

        needs_learned = config.mode == "sparse_learned" or (
            config.mode == "hybrid" and config.sparse_strategy == "learned"
        )
        if needs_learned:
            self._learned_retriever = LearnedSparseRetriever(
                document_store=document_store, config=config.learned_sparse
            )

        if config.mode == "hybrid":
            sparse_sub = self._pick_sparse_retriever(config.sparse_strategy)
            if sparse_sub is None:
                raise ValueError(
                    f"Hybrid mode with sparse_strategy='{config.sparse_strategy}' "
                    "requires an initialized sparse retriever."
                )
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self._dense_retriever,  # type: ignore[arg-type]
                sparse_retriever=sparse_sub,
                fusion_config=config.fusion,
                dense_candidate_k=config.dense.candidate_k,
                sparse_candidate_k=_sparse_candidate_k(config),
            )

        self._reranker = BaselineReranker(config.reranker)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """Run the full retrieval pipeline for a natural language query."""
        if k is None:
            k = self._config.top_k
        if k <= 0:
            return []
        if not query or not query.strip():
            logger.debug("RetrievalOrchestrator.retrieve: empty query → [].")
            return []

        results, _ = self._run_pipeline(query, k)
        return results

    def retrieve_with_trace(self, query: str, k: Optional[int] = None) -> RetrievalTrace:
        """Run the pipeline and return all intermediate data."""
        if k is None:
            k = self._config.top_k
        if k <= 0 or not query or not query.strip():
            return _empty_trace(query, self._config.mode)

        results, trace_data = self._run_pipeline(query, k)
        trace_data["final_results"] = results
        return trace_data  # type: ignore[return-value]

    def _run_pipeline(self, query: str, k: int) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        mode = self._config.mode

        query_tokens: Optional[List[str]] = None
        query_sparse_vec: Optional[Dict[str, float]] = None
        dense_candidates: List[ScoredCandidate] = []
        sparse_candidates: List[ScoredCandidate] = []
        fused_candidates: List[RankedCandidate] = []

        if mode == "dense":
            dense_vec = self._embed_query_dense(query)
            if dense_vec:
                dense_candidates = self._dense_retriever.retrieve_candidates(query=dense_vec, k=k)  # type: ignore[union-attr]
            results = self._join_candidates(dense_candidates)

        elif mode == "sparse_bm25":
            query_tokens = self._tokenizer.tokenize(query)  # type: ignore[union-attr]
            if query_tokens:
                sparse_candidates = self._bm25_retriever.retrieve_candidates(query=query_tokens, k=k)  # type: ignore[union-attr]
            results = self._join_candidates(sparse_candidates)

        elif mode == "sparse_tfidf":
            query_tokens = self._tokenizer.tokenize(query)  # type: ignore[union-attr]
            if query_tokens:
                sparse_candidates = self._tfidf_retriever.retrieve_candidates(query=query_tokens, k=k)  # type: ignore[union-attr]
            results = self._join_candidates(sparse_candidates)

        elif mode == "sparse_learned":
            query_sparse_vec = self._embed_query_sparse(query)
            if query_sparse_vec:
                sparse_candidates = self._learned_retriever.retrieve_candidates(query=query_sparse_vec, k=k)  # type: ignore[union-attr]
            results = self._join_candidates(sparse_candidates)

        elif mode == "hybrid":
            dense_vec = self._embed_query_dense(query)
            sparse_q: Any = self._build_sparse_query(query)

            if isinstance(sparse_q, dict):
                query_sparse_vec = sparse_q
            elif isinstance(sparse_q, list):
                query_tokens = sparse_q

            hybrid_q: HybridQuery = {}
            if dense_vec:
                hybrid_q["dense"] = dense_vec
            if sparse_q is not None:
                hybrid_q["sparse"] = sparse_q

            dense_candidates, sparse_candidates, fused_candidates = (
                self._hybrid_retriever.retrieve_with_details(query=hybrid_q, k=k)  # type: ignore[union-attr]
            )
            results = self._join_ranked(fused_candidates)

        else:
            raise ValueError(
                f"Unknown retrieval mode: '{mode}'. "
                "Valid: 'dense', 'sparse_bm25', 'sparse_tfidf', 'sparse_learned', 'hybrid'."
            )

        results = self._reranker.rerank(query, results)

        logger.debug("RetrievalOrchestrator | mode=%s | results=%d | k=%d", mode, len(results), k)

        trace_data: Dict[str, Any] = {
            "mode": mode,
            "query": query,
            "query_tokens": query_tokens,
            "query_sparse_vector": query_sparse_vec,
            "dense_candidates": dense_candidates,
            "sparse_candidates": sparse_candidates,
            "fused_candidates": fused_candidates,
        }
        return results, trace_data

    def _join_candidates(self, candidates: List[ScoredCandidate]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for candidate in candidates:
            doc = self._doc_by_id.get(candidate["id"])
            if doc is None:
                logger.warning("id '%s' not in DocumentStore — skipped.", candidate["id"])
                continue
            score = candidate["score"]
            results.append({
                "id": doc["id"], "chunk_id": doc["chunk_id"], "document_id": doc["document_id"],
                "score": score, "retrieval_score": score,
                "text": doc["text"], "metadata": doc["metadata"],
            })
        results.sort(key=lambda r: (-r["score"], r["id"]))
        return results

    def _join_ranked(self, fused: List[RankedCandidate]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for candidate in fused:
            doc = self._doc_by_id.get(candidate["id"])
            if doc is None:
                logger.warning("id '%s' not in DocumentStore — skipped.", candidate["id"])
                continue
            score = candidate["score"]
            entry: RetrievalResult = {
                "id": doc["id"], "chunk_id": doc["chunk_id"], "document_id": doc["document_id"],
                "score": score, "retrieval_score": score,
                "text": doc["text"], "metadata": doc["metadata"],
            }
            if candidate.get("dense_score") is not None:
                entry["dense_score"] = candidate["dense_score"]  # type: ignore[typeddict-unknown-key]
            if candidate.get("sparse_score") is not None:
                entry["sparse_score"] = candidate["sparse_score"]  # type: ignore[typeddict-unknown-key]
            results.append(entry)
        results.sort(key=lambda r: (-r["score"], r["id"]))
        return results

    def _embed_query_dense(self, query: str) -> Optional[List[float]]:
        if self._embedder is None:
            return None
        try:
            vecs = self._embedder.embed_queries([query])
            return vecs[0] if vecs and vecs[0] else None
        except Exception as exc:
            logger.warning("Dense query embedding failed: %s", exc)
            return None

    def _embed_query_sparse(self, query: str) -> Optional[Dict[str, float]]:
        if self._embedder is None:
            return None
        try:
            sparse_vecs = self._embedder.embed_queries_sparse([query])
            return sparse_vecs[0] if sparse_vecs and sparse_vecs[0] else None
        except NotImplementedError:
            logger.warning("Embedder does not support sparse output; skipped.")
            return None
        except Exception as exc:
            logger.warning("Sparse query embedding failed: %s", exc)
            return None

    def _build_sparse_query(self, query: str) -> Optional[Any]:
        strategy = self._config.sparse_strategy
        if strategy in {"bm25", "tfidf"}:
            return self._tokenizer.tokenize(query) if self._tokenizer else None
        if strategy == "learned":
            return self._embed_query_sparse(query)
        return None

    def _pick_sparse_retriever(self, strategy: str):
        if strategy == "bm25":
            return self._bm25_retriever
        if strategy == "tfidf":
            return self._tfidf_retriever
        if strategy == "learned":
            return self._learned_retriever
        raise ValueError(f"Unknown sparse strategy: '{strategy}'. Valid: 'bm25', 'tfidf', 'learned'.")

    @classmethod
    def from_index_result(
        cls,
        index_result,
        config: RetrievalConfig,
        embedder: Optional[BaseEmbedder] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> "RetrievalOrchestrator":
        """Construct from an IndexingResult."""
        return cls(
            document_store=index_result.document_store,
            config=config,
            embedder=embedder,
            dense_index=index_result.dense_index,
            sparse_index=index_result.sparse_index,
            tokenizer=tokenizer,
        )


def _validate_dependencies(
    config: RetrievalConfig,
    embedder: Optional[BaseEmbedder],
    dense_index: Optional[BaseDenseIndex],
    sparse_index: Optional[BaseSparseIndex],
    tokenizer: Optional[Tokenizer],
) -> None:
    mode = config.mode

    if mode in _DENSE_MODES and dense_index is None:
        raise ValueError(f"mode='{mode}' requires dense_index.")
    if mode in _DENSE_MODES and embedder is None:
        raise ValueError(f"mode='{mode}' requires embedder for dense query embedding.")

    needs_lexical = mode in _LEXICAL_MODES or (
        mode == "hybrid" and config.sparse_strategy in {"bm25", "tfidf"}
    )
    if needs_lexical and sparse_index is None:
        raise ValueError(f"mode='{mode}' with sparse_strategy='{config.sparse_strategy}' requires sparse_index.")
    if needs_lexical and tokenizer is None:
        raise ValueError(f"mode='{mode}' with sparse_strategy='{config.sparse_strategy}' requires tokenizer.")

    needs_learned = mode in _LEARNED_SPARSE_MODES or (
        mode == "hybrid" and config.sparse_strategy == "learned"
    )
    if needs_learned and embedder is None:
        raise ValueError(f"mode='{mode}' with sparse_strategy='learned' requires an embedder.")


def _sparse_candidate_k(config: RetrievalConfig) -> int:
    if config.sparse_strategy == "bm25":
        return config.bm25.candidate_k
    if config.sparse_strategy == "tfidf":
        return config.tfidf.candidate_k
    if config.sparse_strategy == "learned":
        return config.learned_sparse.candidate_k
    return config.bm25.candidate_k


def _empty_trace(query: str, mode: str) -> RetrievalTrace:
    return {
        "mode": mode, "query": query, "query_tokens": None, "query_sparse_vector": None,
        "dense_candidates": [], "sparse_candidates": [], "fused_candidates": [], "final_results": [],
    }
