"""Tests for IngestionOrchestrator.

The orchestrator is the spine of the ingestion pipeline: load → clean → chunk
→ persist → yield. These tests validate single-pass traversal, deterministic
ordering, fail-fast duplicate detection, metric population, and the
skip_sink integration with loaders that opt in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pytest

from rag.ingestion.cleaner import DefaultCleaner
from rag.ingestion.chunking.base import Chunker
from rag.ingestion.loaders.base import BaseLoader, LoaderResolver
from rag.ingestion.metrics import IngestionMetrics
from rag.ingestion.orchestrator import IngestionOrchestrator
from rag.ingestion.schema import Chunk, Document
from rag.ingestion.storage.base import BaseStore


class StubLoader:
    """Loader yielding canned documents per file path.

    Always accepts ``skip_sink``; the orchestrator passes it whenever a metrics
    object is supplied, so loaders are expected to accept the kwarg.
    """

    def __init__(self, payload_by_name: Dict[str, List[Dict[str, Any]]],
                 accepts_skip_sink: bool = True) -> None:
        # accepts_skip_sink retained for backward compat with tests that pass
        # the flag, but the loader never rejects skip_sink in practice.
        self._payload = payload_by_name
        self._accepts_sink = accepts_skip_sink
        self.received_sinks: List[Any] = []

    def supports(self, path: Path) -> bool:
        return True

    def load(self, path: Path, skip_sink=None):
        self.received_sinks.append(skip_sink)
        for record in self._payload.get(path.name, []):
            yield record


class StubResolver(LoaderResolver):
    def __init__(self, loaders_by_suffix: Dict[str, BaseLoader]) -> None:
        self._loaders = loaders_by_suffix

    def get_loader(self, path: Path) -> BaseLoader:
        loader = self._loaders.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"No loader for {path.suffix}")
        return loader


class FixedChunker(Chunker):
    """Splits content into N equal-size chunks deterministically."""

    def __init__(self, n_chunks: int = 2, strategy_name: str = "test_strategy") -> None:
        self.n_chunks = n_chunks
        self.strategy_name = strategy_name

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        text = doc["content"]
        size = max(1, len(text) // self.n_chunks)
        for i in range(self.n_chunks):
            start = i * size
            end = (i + 1) * size if i < self.n_chunks - 1 else len(text)
            yield {
                "id": f"{doc['id']}_chunk_{i}",
                "document_id": doc["id"],
                "text": text[start:end],
                "metadata": {**doc["metadata"], "chunk_index": i,
                             "strategy": self.strategy_name},
            }


class CollectingStore(BaseStore):
    """In-memory store recording what is persisted."""

    def __init__(self) -> None:
        self.batches: List[List[Dict[str, Any]]] = []

    def write_many(self, items: Iterable[Dict[str, Any]]) -> None:
        self.batches.append(list(items))


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir()
    (d / "b.txt").write_text("ignored", encoding="utf-8")
    (d / "a.txt").write_text("ignored", encoding="utf-8")
    (d / "c.txt").write_text("ignored", encoding="utf-8")
    return d


def _build(loader: StubLoader, **kwargs) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        loader_resolver=StubResolver({".txt": loader}),
        cleaner=DefaultCleaner(),
        chunker=kwargs.get("chunker", FixedChunker()),
        doc_store=kwargs.get("doc_store"),
        chunk_store=kwargs.get("chunk_store"),
        batch_size=kwargs.get("batch_size", 100),
    )


class TestHappyPath:
    def test_load_clean_chunk_yield(self, source_dir: Path) -> None:
        loader = StubLoader({
            "a.txt": [{"id": "n1", "content": "alpha beta\n", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "n2", "content": "gamma delta\n", "metadata": {"source": "b.txt"}}],
            "c.txt": [],
        })
        orch = _build(loader)
        chunks = list(orch.run(source_dir, persist=False))
        assert len(chunks) == 4
        assert all("id" in c and "document_id" in c and "text" in c for c in chunks)

    def test_files_traversed_in_sorted_order(self, source_dir: Path) -> None:
        loader = StubLoader({
            "a.txt": [{"id": "a", "content": "A", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "b", "content": "B", "metadata": {"source": "b.txt"}}],
            "c.txt": [{"id": "c", "content": "C", "metadata": {"source": "c.txt"}}],
        })
        orch = _build(loader, chunker=FixedChunker(n_chunks=1))
        chunks = list(orch.run(source_dir, persist=False))
        order = [c["text"] for c in chunks]
        assert order == ["A", "B", "C"]


class TestDuplicateDetection:
    def test_duplicate_chunk_ids_raise(self, source_dir: Path) -> None:
        class DupChunker(Chunker):
            def chunk(self, doc):
                yield {"id": "same", "document_id": doc["id"], "text": "x",
                       "metadata": {}}

        loader = StubLoader({
            "a.txt": [{"id": "x", "content": "alpha", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "y", "content": "beta", "metadata": {"source": "b.txt"}}],
            "c.txt": [],
        })
        orch = _build(loader, chunker=DupChunker())
        with pytest.raises(ValueError, match="Duplicate chunk ID"):
            list(orch.run(source_dir, persist=False))


class TestSkipSinkIntegration:
    def test_skip_sink_passed_when_loader_accepts_it(self, source_dir: Path) -> None:
        loader = StubLoader(
            {"a.txt": [], "b.txt": [], "c.txt": []},
            accepts_skip_sink=True,
        )
        orch = _build(loader)
        metrics = IngestionMetrics()
        list(orch.run(source_dir, persist=False, metrics=metrics))
        assert len(loader.received_sinks) == 3
        assert all(callable(s) for s in loader.received_sinks)


class TestPersistence:
    def test_batched_chunk_persist(self, source_dir: Path) -> None:
        loader = StubLoader({
            "a.txt": [{"id": "a", "content": "x" * 30, "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "b", "content": "y" * 30, "metadata": {"source": "b.txt"}}],
            "c.txt": [{"id": "c", "content": "z" * 30, "metadata": {"source": "c.txt"}}],
        })
        chunk_store = CollectingStore()
        orch = _build(loader, chunker=FixedChunker(n_chunks=2),
                      chunk_store=chunk_store, batch_size=2)
        out = list(orch.run(source_dir, persist=True))
        # 6 chunks total → batches of 2 + final flush
        assert sum(len(b) for b in chunk_store.batches) == 6
        assert len(out) == 6

    def test_doc_store_persistence(self, source_dir: Path) -> None:
        loader = StubLoader({
            "a.txt": [{"id": "a", "content": "alpha", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "b", "content": "beta", "metadata": {"source": "b.txt"}}],
            "c.txt": [],
        })
        doc_store = CollectingStore()
        orch = _build(loader, chunker=FixedChunker(n_chunks=1),
                      doc_store=doc_store, batch_size=10)
        list(orch.run(source_dir, persist=True))
        total = sum(len(b) for b in doc_store.batches)
        assert total == 2


class TestMetrics:
    def test_metrics_populated(self, source_dir: Path) -> None:
        loader = StubLoader({
            "a.txt": [{"id": "a", "content": "alpha", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "b", "content": "", "metadata": {"source": "b.txt"}}],  # blank → dropped
            "c.txt": [{"id": "c", "content": "gamma", "metadata": {"source": "c.txt"}}],
        })
        orch = _build(loader, chunker=FixedChunker(n_chunks=1))
        metrics = IngestionMetrics()
        list(orch.run(source_dir, persist=False, metrics=metrics))
        assert metrics.docs_loaded == 2
        assert metrics.docs_dropped == 1
        assert metrics.chunks_produced == 2

    def test_files_skipped_counted(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "ok.txt").write_text("alpha", encoding="utf-8")
        (d / "unsupported.xyz").write_text("data", encoding="utf-8")
        loader = StubLoader({"ok.txt": [
            {"id": "n", "content": "alpha", "metadata": {"source": "ok.txt"}}]})
        orch = IngestionOrchestrator(
            loader_resolver=StubResolver({".txt": loader}),
            cleaner=DefaultCleaner(), chunker=FixedChunker(n_chunks=1),
        )
        metrics = IngestionMetrics()
        list(orch.run(d, persist=False, metrics=metrics))
        assert metrics.files_skipped == 1


class TestRecursion:
    def test_walks_subdirectories(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "a.txt").write_text("A", encoding="utf-8")
        (sub / "b.txt").write_text("B", encoding="utf-8")
        loader = StubLoader({
            "a.txt": [{"id": "a", "content": "A", "metadata": {"source": "a.txt"}}],
            "b.txt": [{"id": "b", "content": "B", "metadata": {"source": "b.txt"}}],
        })
        orch = _build(loader, chunker=FixedChunker(n_chunks=1))
        chunks = list(orch.run(d, persist=False))
        assert len(chunks) == 2
