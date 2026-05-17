"""Tests for LoaderRegistry.

The registry is an explicit, extension-based dispatch table. Duplicate
registrations must fail loud; unknown extensions must raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from rag.ingestion.loaders.base import BaseLoader
from rag.ingestion.loaders.registry import LoaderRegistry
from rag.ingestion.schema import Document


class DummyLoader:
    def supports(self, path: Path) -> bool:
        return True

    def load(self, path: Path) -> Iterable[Document]:
        return iter(())


class TestRegistration:
    def test_register_and_get(self) -> None:
        loader = DummyLoader()
        reg = LoaderRegistry()
        reg.register(".md", loader)
        assert reg.get_loader(Path("a.md")) is loader

    def test_case_insensitive_lookup(self) -> None:
        loader = DummyLoader()
        reg = LoaderRegistry()
        reg.register(".md", loader)
        assert reg.get_loader(Path("A.MD")) is loader

    def test_duplicate_registration_raises(self) -> None:
        reg = LoaderRegistry()
        reg.register(".md", DummyLoader())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(".md", DummyLoader())

    def test_invalid_extension_raises(self) -> None:
        reg = LoaderRegistry()
        with pytest.raises(ValueError, match="must start with"):
            reg.register("md", DummyLoader())

    def test_unknown_extension_raises(self) -> None:
        reg = LoaderRegistry()
        with pytest.raises(ValueError, match="No loader registered"):
            reg.get_loader(Path("a.xyz"))


class TestBulkRegistration:
    def test_register_many(self) -> None:
        a = DummyLoader()
        b = DummyLoader()
        reg = LoaderRegistry(loaders=[(".md", a), (".jsonl", b)])
        assert reg.get_loader(Path("x.md")) is a
        assert reg.get_loader(Path("x.jsonl")) is b

    def test_supported_extensions_sorted(self) -> None:
        reg = LoaderRegistry(loaders=[(".md", DummyLoader()),
                                      (".jsonl", DummyLoader())])
        assert reg.supported_extensions() == sorted(reg.supported_extensions())
