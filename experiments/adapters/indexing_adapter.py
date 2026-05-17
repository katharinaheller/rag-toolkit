"""Adapter for ``rag.indexing``.

Builds and caches indexes per ``(retriever, corpus)`` combination. The dense
backend is chosen automatically by corpus size: brute force on small corpora,
FAISS on larger ones when available.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rag.indexing.config import (
    DenseIndexConfig,
    FAISSConfig,
    IndexConfig,
    SparseIndexConfig,
)
from rag.indexing.orchestrator import IndexingOrchestrator, IndexingResult
from rag.indexing.sparse.tokenizer import Tokenizer

from experiments.adapters.embedding_adapter import embed_corpus
from experiments.configs.default_matrix import EmbedderSpec, RetrieverSpec
from experiments.configs.settings import SETTINGS

logger = logging.getLogger(__name__)


try:
    import faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


@dataclass
class BuiltIndex:
    """Bundles the IndexingResult with the tokenizer + build timing."""

    index_result: IndexingResult
    tokenizer: Tokenizer
    config: IndexConfig
    build_time_s: float
    document_count: int


def _index_dir_for(retriever_key: str, corpus_name: str) -> Path:
    return SETTINGS.cache_root / "indexes" / retriever_key / corpus_name


def _pick_dense_backend(n_chunks: int) -> str:
    """Use brute_force for small corpora, FAISS Flat for large ones if available."""
    if not _FAISS_AVAILABLE:
        return "brute_force"
    return "brute_force" if n_chunks < 500 else "faiss"


def _make_index_config(
    retriever: RetrieverSpec,
    embedder: Optional[EmbedderSpec],
    n_chunks: int,
    index_dir: Path,
) -> IndexConfig:
    mode_for_index = "dense"
    if retriever.needs_dense and retriever.needs_sparse_lexical:
        mode_for_index = "hybrid"
    elif retriever.needs_dense:
        mode_for_index = "dense"
    elif retriever.needs_sparse_lexical:
        mode_for_index = "sparse"
    else:
        mode_for_index = "sparse"  # fallback

    dense_cfg = DenseIndexConfig(
        backend=_pick_dense_backend(n_chunks),
        metric="cosine",
        dimension=embedder.dimension if embedder else None,
        faiss=FAISSConfig(index_type="flat"),
    )
    sparse_cfg = SparseIndexConfig(tokenizer="simple")

    return IndexConfig(
        index_dir=index_dir,
        mode=mode_for_index,
        dense=dense_cfg,
        sparse=sparse_cfg,
    )


def build_index(
    retriever: RetrieverSpec,
    embedder: Optional[EmbedderSpec],
    chunks: List[Dict],
    corpus_name: str,
    force: bool = False,
) -> BuiltIndex:
    """Construct the index for one retriever × corpus.

    Embeddings are produced (cached) if the retriever needs them.
    """
    index_dir = _index_dir_for(retriever.key, corpus_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    marker = index_dir / ".built"

    cfg = _make_index_config(retriever, embedder, len(chunks), index_dir)
    tokenizer = Tokenizer(cfg.sparse)

    # Resolve embedding mode for embedder.
    embeddings: List[Dict]
    if retriever.needs_dense and embedder is not None:
        emb_mode = "hybrid" if (retriever.needs_dense
                                and retriever.needs_sparse_lexical
                                and embedder.supports_sparse
                                and retriever.sparse_strategy == "learned") else "dense"
        embeddings = embed_corpus(embedder, chunks, corpus_name, mode=emb_mode, force=force)
    else:
        # Sparse-only retrievers still need at least chunk metadata threaded
        # through the orchestrator; we fabricate minimal embedding records
        # carrying just id/chunk_id/document_id (no vectors).
        embeddings = [
            {
                "id": f"sparse_{chunk['id']}",
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "embedding": None,
                "sparse_embedding": None,
                "embedding_type": "sparse",
                "model_type": "lexical",
                "projection_method": "none",
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ]

    if marker.exists() and not force:
        try:
            orchestrator = IndexingOrchestrator(cfg)
            t0 = time.perf_counter()
            index_result = orchestrator.load()
            dt = time.perf_counter() - t0
            logger.info(
                "Loaded cached index %s × %s in %.2fs", retriever.key, corpus_name, dt,
            )
            return BuiltIndex(
                index_result=index_result,
                tokenizer=tokenizer,
                config=cfg,
                build_time_s=dt,
                document_count=len(chunks),
            )
        except Exception as exc:
            logger.warning(
                "Cached index load failed for %s × %s (%s); rebuilding",
                retriever.key, corpus_name, exc,
            )

    orchestrator = IndexingOrchestrator(cfg)
    t0 = time.perf_counter()
    index_result = orchestrator.build(
        embeddings=embeddings,
        chunks=chunks,
        persist=True,
    )
    dt = time.perf_counter() - t0
    marker.write_text(json.dumps({
        "retriever": retriever.key,
        "corpus": corpus_name,
        "n_chunks": len(chunks),
        "build_time_s": dt,
    }), encoding="utf-8")

    logger.info(
        "Built index %s × %s in %.2fs (n_chunks=%d)",
        retriever.key, corpus_name, dt, len(chunks),
    )

    return BuiltIndex(
        index_result=index_result,
        tokenizer=tokenizer,
        config=cfg,
        build_time_s=dt,
        document_count=len(chunks),
    )
