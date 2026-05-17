from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from rag.ingestion.schema import Document


@runtime_checkable
class BaseLoader(Protocol):
    """Contract: determine support for a path and load Documents from it."""

    def supports(self, path: Path) -> bool:
        ...

    def load(self, path: Path) -> Iterable[Document]:
        ...


@runtime_checkable
class LoaderResolver(Protocol):
    """Contract: map a file path to its registered loader."""

    def get_loader(self, path: Path) -> BaseLoader:
        ...
