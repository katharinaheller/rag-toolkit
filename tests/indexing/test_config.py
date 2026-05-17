"""Validation tests for the indexing configuration dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.config import (
    DenseIndexConfig,
    FAISSConfig,
    IndexConfig,
    SparseIndexConfig,
)


class TestFAISSConfig:
    def test_defaults(self) -> None:
        c = FAISSConfig()
        assert c.index_type == "flat"

    @pytest.mark.parametrize("t", ["flat", "hnsw", "ivf"])
    def test_accepts_known_types(self, t: str) -> None:
        FAISSConfig(index_type=t)

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="index_type"):
            FAISSConfig(index_type="lsh")

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_search_k_factor(self, k: int) -> None:
        with pytest.raises(ValueError, match="search_k_factor"):
            FAISSConfig(search_k_factor=k)


class TestDenseIndexConfig:
    def test_defaults_are_brute_force_cosine(self) -> None:
        c = DenseIndexConfig()
        assert c.backend == "brute_force"
        assert c.metric == "cosine"

    @pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
    def test_accepts_known_metrics(self, metric: str) -> None:
        DenseIndexConfig(metric=metric)

    def test_rejects_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            DenseIndexConfig(metric="manhattan")

    @pytest.mark.parametrize("backend", ["brute_force", "faiss"])
    def test_accepts_known_backends(self, backend: str) -> None:
        DenseIndexConfig(backend=backend)

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            DenseIndexConfig(backend="annoy")

    @pytest.mark.parametrize("d", [0, -1])
    def test_rejects_non_positive_dimension(self, d: int) -> None:
        with pytest.raises(ValueError, match="dimension"):
            DenseIndexConfig(dimension=d)


class TestSparseIndexConfig:
    @pytest.mark.parametrize("tok", ["simple", "whitespace"])
    def test_accepts_known_tokenizers(self, tok: str) -> None:
        SparseIndexConfig(tokenizer=tok)

    def test_rejects_unknown_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            SparseIndexConfig(tokenizer="bert")


class TestIndexConfig:
    def test_default_dense_mode(self, tmp_path: Path) -> None:
        c = IndexConfig(index_dir=tmp_path)
        assert c.mode == "dense"

    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
    def test_accepts_known_modes(self, mode: str, tmp_path: Path) -> None:
        IndexConfig(index_dir=tmp_path, mode=mode)

    def test_rejects_unknown_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            IndexConfig(index_dir=tmp_path, mode="weird")
