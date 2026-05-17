from pathlib import Path
from typing import Dict, Iterable, Tuple

from rag.ingestion.loaders.base import BaseLoader


class LoaderRegistry:
    """Extension-based loader registry.

    No implicit registrations, no content inspection. Duplicate registration
    fails immediately at startup.
    """

    def __init__(self, loaders: Iterable[Tuple[str, BaseLoader]] | None = None) -> None:
        self._loaders: Dict[str, BaseLoader] = {}
        if loaders is not None:
            self.register_many(loaders)

    def register(self, extension: str, loader: BaseLoader) -> None:
        ext = extension.lower()
        if not ext.startswith("."):
            raise ValueError(f"Extension must start with '.', got: {ext!r}")
        if ext in self._loaders:
            raise ValueError(f"Loader already registered for extension: {ext!r}")
        self._loaders[ext] = loader

    def register_many(self, loaders: Iterable[Tuple[str, BaseLoader]]) -> None:
        for extension, loader in loaders:
            self.register(extension, loader)

    def get_loader(self, path: Path) -> BaseLoader:
        ext = path.suffix.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise ValueError(f"No loader registered for extension: {ext!r}")
        return loader

    def supported_extensions(self) -> list[str]:
        return sorted(self._loaders.keys())
