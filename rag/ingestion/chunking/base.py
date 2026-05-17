from typing import Iterable, Protocol, runtime_checkable

from rag.ingestion.schema import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Contract: split a Document into an ordered, finite sequence of Chunks."""

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        ...
