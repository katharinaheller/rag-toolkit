from dataclasses import dataclass
from pathlib import Path

from rag.ingestion.chunking.base import Chunker
from rag.ingestion.cleaner import Cleaner
from rag.ingestion.loaders.base import BaseLoader
from rag.ingestion.storage.base import BaseStore


@dataclass(frozen=True)
class LoaderBinding:
    """Maps a file extension (e.g. '.txt') to its loader."""
    extension: str
    loader: BaseLoader


@dataclass(frozen=True)
class IngestionComponents:
    """Complete, immutable set of components for a pipeline run."""
    loader_bindings: tuple[LoaderBinding, ...]
    cleaner: Cleaner
    chunker: Chunker
    doc_store: BaseStore | None
    chunk_store: BaseStore | None


@dataclass(frozen=True)
class IngestionRequest:
    """A single ingestion job: source path + persist flag."""
    source: Path
    persist: bool
