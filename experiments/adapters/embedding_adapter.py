"""Adapter for ``rag.embedding``.

Builds an embedder from an :class:`EmbedderSpec` and computes embedding vectors
for chunks / queries through the real ``EmbeddingOrchestrator``. Embeddings are
cached on disk per (embedder_key, corpus_name) so subsequent suites reuse them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rag.embedding.config import (
    EmbeddingBehaviorConfig,
    EmbeddingConfig,
    EmbeddingProjectionConfig,
)
from rag.embedding.factory import create_embedder
from rag.embedding.orchestrator import EmbeddingOrchestrator

from experiments.configs.default_matrix import EmbedderSpec
from experiments.configs.settings import SETTINGS

logger = logging.getLogger(__name__)

_EMBEDDER_INSTANCE_CACHE: Dict[str, object] = {}


def _embedder_cache_key(spec: EmbedderSpec, mode: str) -> str:
    return f"{spec.key}::{mode}"


def _build_config(spec: EmbedderSpec, mode: str, behavior_mode: str) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=spec.provider,
        model_name=spec.model_name,
        device="cpu",
        batch_size=spec.batch_size,
        max_seq_length=spec.max_seq_length,
        use_fp16=spec.use_fp16,
        use_bfloat16=spec.use_bfloat16,
        retrieval_mode=mode,
        behavior=EmbeddingBehaviorConfig(
            normalize=True,
            mode=behavior_mode,
        ),
        projection=EmbeddingProjectionConfig(target_dim=None),
    )


def get_embedder(spec: EmbedderSpec, mode: str = "dense"):
    """Return a cached embedder instance for the given spec + mode."""
    key = _embedder_cache_key(spec, mode)
    cached = _EMBEDDER_INSTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    cfg = _build_config(spec, mode=mode, behavior_mode="document")
    logger.info(
        "Building embedder spec=%s provider=%s model=%s mode=%s",
        spec.key, spec.provider, spec.model_name, mode,
    )
    embedder = create_embedder(cfg)
    _EMBEDDER_INSTANCE_CACHE[key] = embedder
    return embedder


def _embeddings_cache_path(spec: EmbedderSpec, corpus_name: str, mode: str) -> Path:
    cache_dir = SETTINGS.cache_root / "embeddings" / spec.key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{corpus_name}_{mode}.jsonl"


def _chunks_digest(chunks: List[Dict]) -> str:
    """Stable digest of chunk-ids so cache invalidates if chunking changes."""
    h = hashlib.sha1()
    for chunk in chunks:
        h.update(chunk["id"].encode("utf-8"))
    return h.hexdigest()[:12]


def embed_corpus(
    spec: EmbedderSpec,
    chunks: List[Dict],
    corpus_name: str,
    mode: str = "dense",
    force: bool = False,
) -> List[Dict]:
    """Embed all corpus chunks. Returns ``EmbeddingVector`` dicts.

    Caching uses both the embedder key and a chunk-id digest so the cache
    invalidates automatically when chunking parameters change.
    """
    cache_path = _embeddings_cache_path(spec, corpus_name, mode)
    digest_path = cache_path.with_suffix(".digest")
    current_digest = _chunks_digest(chunks)

    if cache_path.exists() and digest_path.exists() and not force:
        if digest_path.read_text(encoding="utf-8").strip() == current_digest:
            logger.info("Using cached embeddings: %s", cache_path)
            with cache_path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        else:
            logger.info("Cache digest mismatch for %s — re-embedding", cache_path)

    embedder = get_embedder(spec, mode=mode)
    cfg = _build_config(spec, mode=mode, behavior_mode="document")
    orchestrator = EmbeddingOrchestrator(embedder=embedder, config=cfg, store=None)

    logger.info(
        "Embedding %d chunks with %s (mode=%s) → %s",
        len(chunks), spec.key, mode, cache_path,
    )
    vectors = orchestrator.run(chunks, persist=False)

    with cache_path.open("w", encoding="utf-8") as f:
        for vec in vectors:
            f.write(json.dumps(vec, ensure_ascii=False) + "\n")
    digest_path.write_text(current_digest, encoding="utf-8")

    return vectors


def embed_queries_dense(
    spec: EmbedderSpec, queries: List[str]
) -> List[List[float]]:
    """Embed query strings (no caching: query lists are usually unique)."""
    embedder = get_embedder(spec, mode="dense")
    return embedder.embed_queries(queries)


def supports_sparse(spec: EmbedderSpec) -> bool:
    return spec.supports_sparse
