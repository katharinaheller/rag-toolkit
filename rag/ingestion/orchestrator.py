from pathlib import Path
from typing import Iterator

from rag.ingestion.schema import Chunk, Document
from rag.ingestion.ids import resolve_document_id
from rag.ingestion.cleaner import Cleaner
from rag.ingestion.chunking.base import Chunker
from rag.ingestion.loaders.base import LoaderResolver
from rag.ingestion.metrics import IngestionMetrics
from rag.ingestion.storage.base import BaseStore
from rag.logging.logger import get_logger

logger = get_logger(__name__)


class IngestionOrchestrator:
    """Single-pass streaming pipeline: load → clean → chunk → (persist) → yield.

    Invariants:
        - Source is traversed exactly once regardless of persist flag.
        - Every stage is a generator; no stage materializes the full dataset.
        - File discovery uses sorted() for deterministic ordering.
        - Duplicate chunk IDs raise immediately (fail fast over silent corruption).
        - If a loader accepts skip_sink, the orchestrator passes one that
          increments metrics.records_skipped.
    """

    def __init__(
        self,
        loader_resolver: LoaderResolver,
        cleaner: Cleaner,
        chunker: Chunker,
        doc_store: BaseStore | None = None,
        chunk_store: BaseStore | None = None,
        batch_size: int = 100,
    ) -> None:
        self.loader_resolver = loader_resolver
        self.cleaner = cleaner
        self.chunker = chunker
        self.doc_store = doc_store
        self.chunk_store = chunk_store
        self.batch_size = batch_size

    def _iter_files(self, source: Path) -> Iterator[Path]:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                yield path

    def _load_documents(self, source: Path, metrics: IngestionMetrics | None) -> Iterator[Document]:
        sink = metrics.skip_sink() if metrics is not None else None

        for path in self._iter_files(source):
            try:
                loader = self.loader_resolver.get_loader(path)
            except ValueError:
                if metrics is not None:
                    metrics.files_skipped += 1
                continue

            if sink is not None:
                try:
                    raw_iter = loader.load(path, skip_sink=sink)
                except TypeError:
                    raw_iter = loader.load(path)
            else:
                raw_iter = loader.load(path)

            for raw in raw_iter:
                yield Document(id=raw.get("id", ""), content=raw["content"], metadata=raw["metadata"])

    def _clean_documents(self, docs: Iterator[Document], metrics: IngestionMetrics | None) -> Iterator[Document]:
        """Normalize text and assign deterministic document IDs."""
        for doc in docs:
            cleaned = self.cleaner.clean(doc["content"])

            if cleaned is None:
                if metrics is not None:
                    metrics.docs_dropped += 1
                continue

            metadata = doc["metadata"]
            source = metadata.get("source", "")
            line = metadata.get("line")

            if source:
                natural_key = f"{source}:{line}" if line is not None else source
            else:
                natural_key = doc.get("id", "")

            document_id = resolve_document_id(raw_id=natural_key, cleaned_text=cleaned)
            assert document_id, "document_id must never be empty after resolution"

            if metrics is not None:
                metrics.docs_loaded += 1

            yield Document(id=document_id, content=cleaned, metadata=metadata)

    def _chunk_document(self, doc: Document) -> Iterator[Chunk]:
        yield from self.chunker.chunk(doc)

    def run(self, source: Path, persist: bool, metrics: IngestionMetrics | None = None) -> Iterator[Chunk]:
        logger.info("rag.ingestion.start", source=str(source), persist=persist)

        seen_chunk_ids: set[str] = set()
        doc_batch: list[Document] = []
        chunk_batch: list[Chunk] = []

        for doc in self._clean_documents(self._load_documents(source, metrics), metrics):
            if persist and self.doc_store is not None:
                doc_batch.append(doc)
                if len(doc_batch) >= self.batch_size:
                    self.doc_store.write_many(doc_batch)
                    doc_batch.clear()

            for chunk in self._chunk_document(doc):
                chunk_id = chunk["id"]

                if chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk_id!r} (document_id={chunk['document_id']!r})"
                    )
                seen_chunk_ids.add(chunk_id)

                if metrics is not None:
                    metrics.chunks_produced += 1

                if persist and self.chunk_store is not None:
                    chunk_batch.append(chunk)
                    if len(chunk_batch) >= self.batch_size:
                        self.chunk_store.write_many(chunk_batch)
                        yield from chunk_batch
                        chunk_batch.clear()
                else:
                    yield chunk

        if persist:
            if self.doc_store is not None and doc_batch:
                self.doc_store.write_many(doc_batch)
            if self.chunk_store is not None and chunk_batch:
                self.chunk_store.write_many(chunk_batch)
                yield from chunk_batch

        if metrics is not None:
            metrics.log(logger)

        logger.info("rag.ingestion.done")
