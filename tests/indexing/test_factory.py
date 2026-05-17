"""Tests for the indexing factory functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.backends.brute_force import BruteForceIndex
from rag.indexing.config import DenseIndexConfig, IndexConfig
from rag.indexing.factory import create_dense_index, create_sparse_index
from rag.indexing.sparse.sparse_index import SparseIndex


def test_creates_brute_force_when_configured(tmp_path: Path) -> None:
    cfg = IndexConfig(index_dir=tmp_path, dense=DenseIndexConfig(backend="brute_force"))
    idx = create_dense_index(cfg)
    assert isinstance(idx, BruteForceIndex)


@pytest.mark.requires_faiss
def test_creates_faiss_index_when_available(tmp_path: Path) -> None:
    from rag.indexing.backends.faiss_index import FAISSIndex
    cfg = IndexConfig(index_dir=tmp_path, dense=DenseIndexConfig(backend="faiss"))
    idx = create_dense_index(cfg)
    assert isinstance(idx, FAISSIndex)


def test_creates_sparse_index(tmp_path: Path) -> None:
    cfg = IndexConfig(index_dir=tmp_path)
    idx = create_sparse_index(cfg)
    assert isinstance(idx, SparseIndex)
