"""Adapter for ``rag.retrieval``.

Builds a configured :class:`RetrievalOrchestrator` from an
:class:`experiments.adapters.indexing_adapter.BuiltIndex` and exposes a thin
``retrieve_timed`` wrapper that records per-query latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rag.retrieval.config import (
    BM25Config,
    DenseRetrievalConfig,
    FusionConfig,
    RetrievalConfig,
    TFIDFConfig,
    RerankerConfig,
)
from rag.retrieval.orchestrator import RetrievalOrchestrator

from experiments.adapters.embedding_adapter import get_embedder
from experiments.adapters.indexing_adapter import BuiltIndex
from experiments.configs.default_matrix import (
    EmbedderSpec,
    RetrieverSpec,
    EMBEDDERS_BY_KEY,
)

logger = logging.getLogger(__name__)


@dataclass
class BuiltRetriever:
    """A fully wired retriever, ready to call ``retrieve`` on."""

    orchestrator: RetrievalOrchestrator
    retriever_key: str
    corpus_name: str
    top_k: int
    embedder_spec: Optional[EmbedderSpec]


def build_retrieval_config(
    retriever: RetrieverSpec,
    top_k: int,
    fusion_dense_weight: Optional[float] = None,
    fusion_sparse_weight: Optional[float] = None,
) -> RetrievalConfig:
    """Translate a :class:`RetrieverSpec` into ``rag.retrieval`` config."""
    dw = fusion_dense_weight if fusion_dense_weight is not None else retriever.fusion_dense_weight
    sw = fusion_sparse_weight if fusion_sparse_weight is not None else retriever.fusion_sparse_weight

    return RetrievalConfig(
        mode=retriever.mode,
        sparse_strategy=retriever.sparse_strategy,
        top_k=top_k,
        dense=DenseRetrievalConfig(candidate_k=max(top_k * 4, 50)),
        bm25=BM25Config(candidate_k=max(top_k * 4, 50)),
        tfidf=TFIDFConfig(candidate_k=max(top_k * 4, 50)),
        fusion=FusionConfig(
            strategy="weighted_sum",
            dense_weight=dw,
            sparse_weight=sw,
            combination="union",
            normalize_scores=True,
        ),
        reranker=RerankerConfig(enabled=False),
    )


def build_retriever(
    retriever: RetrieverSpec,
    index: BuiltIndex,
    top_k: int,
    fusion_dense_weight: Optional[float] = None,
    fusion_sparse_weight: Optional[float] = None,
) -> BuiltRetriever:
    """Create a :class:`BuiltRetriever`.

    ``BuiltIndex`` carries the document store, indexes, and tokenizer; this
    factory adds the embedder (when needed) and the retrieval config.
    """
    config = build_retrieval_config(
        retriever, top_k, fusion_dense_weight, fusion_sparse_weight
    )

    embedder_spec: Optional[EmbedderSpec] = None
    embedder = None
    if retriever.embedder_key is not None:
        embedder_spec = EMBEDDERS_BY_KEY[retriever.embedder_key]
        if retriever.needs_dense:
            embedder = get_embedder(embedder_spec, mode="dense")

    orchestrator = RetrievalOrchestrator.from_index_result(
        index_result=index.index_result,
        config=config,
        embedder=embedder,
        tokenizer=index.tokenizer if retriever.needs_sparse_lexical else None,
    )

    return BuiltRetriever(
        orchestrator=orchestrator,
        retriever_key=retriever.key,
        corpus_name=index.config.index_dir.parent.name,
        top_k=top_k,
        embedder_spec=embedder_spec,
    )


def retrieve_timed(
    built: BuiltRetriever, query: str, k: Optional[int] = None
) -> Tuple[List[Dict], float]:
    """Run a retrieval call and return ``(results, elapsed_ms)``."""
    t0 = time.perf_counter()
    try:
        results = built.orchestrator.retrieve(query, k=k or built.top_k)
    except Exception as exc:
        logger.warning(
            "Retrieval failed (retriever=%s, query=%r): %s",
            built.retriever_key, query[:80], exc,
        )
        results = []
    elapsed_ms = (time.perf_counter() - t0) * 1_000.0
    return results, elapsed_ms
