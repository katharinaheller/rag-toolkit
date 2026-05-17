"""Shared per-suite retrieval routines.

Every quality-related suite needs the same primitive: "run retriever R over
query set Q on corpus C and collect per-query retrieval records". Putting that
primitive here keeps each suite under a hundred lines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from experiments.adapters.indexing_adapter import BuiltIndex, build_index
from experiments.adapters.retrieval_adapter import (
    BuiltRetriever,
    build_retriever,
    retrieve_timed,
)
from experiments.configs.default_matrix import (
    EMBEDDERS_BY_KEY,
    EmbedderSpec,
    RetrieverSpec,
)
from experiments.core.types import RetrievalRecord, SyntheticQuery

logger = logging.getLogger(__name__)


def resolve_embedder(retriever: RetrieverSpec) -> Optional[EmbedderSpec]:
    if retriever.embedder_key is None:
        return None
    return EMBEDDERS_BY_KEY[retriever.embedder_key]


def get_or_build_index(
    retriever: RetrieverSpec,
    chunks: List[Dict],
    corpus_name: str,
    cache: Dict[Tuple[str, str], BuiltIndex],
) -> BuiltIndex:
    cache_key = (retriever.key, corpus_name)
    if cache_key in cache:
        return cache[cache_key]
    embedder = resolve_embedder(retriever)
    built = build_index(retriever, embedder, chunks, corpus_name=corpus_name)
    cache[cache_key] = built
    return built


def run_retriever_on_queries(
    run_id: str,
    suite_key: str,
    retriever: RetrieverSpec,
    built_index: BuiltIndex,
    corpus_name: str,
    queries: List[SyntheticQuery],
    top_k: int,
    repeats: int = 1,
    fusion_dense_weight: Optional[float] = None,
    fusion_sparse_weight: Optional[float] = None,
) -> List[RetrievalRecord]:
    """Run retrieval over a query set and return per-rank records.

    ``repeats`` re-runs each query (useful for stability/jitter analysis).
    """
    embedder_spec = resolve_embedder(retriever)
    built = build_retriever(
        retriever, built_index, top_k=top_k,
        fusion_dense_weight=fusion_dense_weight,
        fusion_sparse_weight=fusion_sparse_weight,
    )

    out: List[RetrievalRecord] = []
    for repeat in range(repeats):
        for query in queries:
            results, latency_ms = retrieve_timed(built, query.text, k=top_k)
            for rank, hit in enumerate(results):
                out.append(RetrievalRecord(
                    run_id=run_id,
                    suite=suite_key,
                    retriever=retriever.key,
                    embedder=embedder_spec.key if embedder_spec else None,
                    corpus=corpus_name,
                    top_k=top_k,
                    query_id=query.query_id,
                    query_text=query.text,
                    query_type=query.query_type,
                    rank=rank,
                    chunk_id=hit.get("chunk_id", ""),
                    document_id=hit.get("document_id", ""),
                    score=float(hit.get("score", 0.0)),
                    text_excerpt=(hit.get("text", "") or "")[:240],
                    latency_ms=latency_ms,
                    repeat_index=repeat,
                ))
    return out


def gold_lookup_chunks(queries: List[SyntheticQuery]) -> Dict[str, set]:
    return {q.query_id: set(q.relevant_chunk_ids) for q in queries}


def gold_lookup_documents(queries: List[SyntheticQuery]) -> Dict[str, set]:
    return {q.query_id: set(q.relevant_document_ids) for q in queries}
