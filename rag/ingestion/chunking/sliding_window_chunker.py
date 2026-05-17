from typing import Iterable

from rag.ingestion.schema import Chunk, Document
from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.core import build_chunk
from rag.ingestion.chunking.strategies import ChunkingStrategy


class SlidingWindowChunker(Chunker):
    """Splits text into overlapping fixed-size character windows."""

    def __init__(self, chunk_size: int, overlap: int, strategy: ChunkingStrategy) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        text = doc["content"]
        step = self.chunk_size - self.overlap
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            yield build_chunk(
                document_id=doc["id"],
                index=idx,
                text=text[start:end],
                base_metadata=doc["metadata"],
                extra_metadata={"char_start": start, "char_end": end},
                strategy=self.strategy,
            )
            start += step
            idx += 1
