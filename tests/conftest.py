"""Top-level pytest fixtures and configuration for the RAG test suite.

Shared fixtures, optional-dependency skip markers, deterministic seeding,
and isolation guarantees live here so that every nested test module inherits
the same baseline environment.
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


# Ensure deterministic Python hashing in subprocess scenarios.
os.environ.setdefault("PYTHONHASHSEED", "0")


def _has_module(name: str) -> bool:
    """Return True if a module can be imported without side effects."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


_HAS_FAISS = _has_module("faiss")
_HAS_TORCH = _has_module("torch")
_HAS_ST = _has_module("sentence_transformers")
_HAS_FLAG = _has_module("FlagEmbedding")
_HAS_MARKO = _has_module("marko")
_HAS_PSUTIL = _has_module("psutil")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests whose optional dependency is unavailable."""
    skip_map = {
        "requires_faiss": (_HAS_FAISS, "FAISS not installed"),
        "requires_torch": (_HAS_TORCH, "torch not installed"),
        "requires_sentence_transformers": (_HAS_ST, "sentence-transformers not installed"),
        "requires_flag_embedding": (_HAS_FLAG, "FlagEmbedding not installed"),
        "requires_marko": (_HAS_MARKO, "marko not installed"),
        "requires_psutil": (_HAS_PSUTIL, "psutil not installed"),
    }
    for item in items:
        for marker, (available, reason) in skip_map.items():
            if marker in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _seed_random() -> None:
    """Reseed Python's PRNG before each test for cross-test determinism."""
    random.seed(0)


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Disposable directory for index/store artefacts in a single test."""
    d = tmp_path / "index"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    """Path for a JSONL store; parent created, file not yet present."""
    return tmp_path / "store.jsonl"


@pytest.fixture(autouse=True)
def _silence_structlog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent structlog from polluting captured output during tests."""
    logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture
def data_dir() -> Path:
    """Path to the bundled sample data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def isolated_module_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Snapshot sys.modules and restore on teardown.

    Useful for tests that monkeypatch optional dependency import flags.
    """
    snapshot = dict(sys.modules)
    yield
    for name in list(sys.modules.keys()):
        if name not in snapshot:
            del sys.modules[name]
