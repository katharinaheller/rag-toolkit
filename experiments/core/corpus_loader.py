"""Load and chunk a corpus through the real ``rag.ingestion`` pipeline.

We never re-implement loading or cleaning here: we configure the existing
components and run them. Outputs are cached to disk so subsequent suites do
not pay the chunking cost again.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP
from rag.ingestion.cleaner import DefaultCleaner
from rag.ingestion.composition import (
    IngestionComponents,
    IngestionRequest,
    LoaderBinding,
)
from rag.ingestion.ingestion_api import create_ingestion_service, run_ingestion
from rag.ingestion.loaders.md_loader import MdLoader
from rag.ingestion.metrics import IngestionMetrics

from experiments.configs.settings import SETTINGS, corpus_dir

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _chunks_cache_path(corpus_name: str) -> Path:
    cache_dir = SETTINGS.cache_root / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{corpus_name}_chunks.jsonl"


def load_corpus_chunks(corpus_name: str, force: bool = False) -> List[Dict]:
    """Load and chunk a corpus, caching the result on disk.

    Returns a list of ``Chunk`` dicts with keys ``id``, ``document_id``,
    ``text``, ``metadata``.
    """
    cache_path = _chunks_cache_path(corpus_name)
    if cache_path.exists() and not force:
        logger.info("Using cached chunks for %s from %s", corpus_name, cache_path)
        with cache_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    source = corpus_dir(corpus_name)
    if not source.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {source}. "
            "Set EXPERIMENTS_DATA_ROOT or place corpora under data/documents/."
        )

    logger.info("Ingesting corpus %s from %s", corpus_name, source)

    components = IngestionComponents(
        loader_bindings=(LoaderBinding(extension=".md", loader=MdLoader()),),
        cleaner=DefaultCleaner(),
        chunker=SlidingWindowChunker(
            chunk_size=_CHUNK_SIZE,
            overlap=_CHUNK_OVERLAP,
            strategy=FIXED_OVERLAP,
        ),
        doc_store=None,
        chunk_store=None,
    )
    service = create_ingestion_service(components)
    request = IngestionRequest(source=source, persist=False)
    metrics = IngestionMetrics()

    chunks: List[Dict] = list(run_ingestion(service, request, metrics=metrics))

    if not chunks:
        raise RuntimeError(
            f"Ingestion produced no chunks for corpus {corpus_name}. "
            f"docs_loaded={metrics.docs_loaded}, files_skipped={metrics.files_skipped}."
        )

    logger.info(
        "Ingested %s: docs_loaded=%d chunks=%d docs_dropped=%d files_skipped=%d",
        corpus_name,
        metrics.docs_loaded,
        len(chunks),
        metrics.docs_dropped,
        metrics.files_skipped,
    )

    # Persist to cache.
    with cache_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return chunks


def chunks_by_id(chunks: List[Dict]) -> Dict[str, Dict]:
    return {c["id"]: c for c in chunks}


def chunks_by_document(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk["document_id"], []).append(chunk)
    return grouped
