"""Cross-cutting configuration validation tests.

Each component already has its own targeted test suite; this module asserts
that the *system-level* invariants spanning multiple configs remain stable:
- mutually exclusive flags
- positive numerical constraints
- enum-style allowed-values
- composition: hybrid retrieval depends on consistent sparse_strategy
"""

from __future__ import annotations

import pytest

from rag.embedding.config import (
    EmbeddingBehaviorConfig,
    EmbeddingConfig,
    EmbeddingProjectionConfig,
)
from rag.generation.config import GenerationConfig
from rag.indexing.config import (
    DenseIndexConfig,
    FAISSConfig,
    IndexConfig,
    SparseIndexConfig,
)
from rag.retrieval.config import (
    BM25Config,
    DenseRetrievalConfig,
    FusionConfig,
    LearnedSparseConfig,
    RerankerConfig,
    RetrievalConfig,
    TFIDFConfig,
)


class TestEmbeddingConfigSafety:
    def test_fp16_and_bfloat16_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            EmbeddingConfig(
                provider="x", model_name="m", use_bfloat16=True, use_fp16=True,
            )

    def test_batch_size_positive(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="x", model_name="m", batch_size=0)

    def test_max_seq_length_positive_when_set(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="x", model_name="m", max_seq_length=0)

    def test_invalid_retrieval_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="x", model_name="m", retrieval_mode="bogus")

    def test_projection_target_dim_positive(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingProjectionConfig(target_dim=0)

    def test_projection_method_enum(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingProjectionConfig(method="bogus")

    def test_behavior_mode_enum(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingBehaviorConfig(mode="bogus")


class TestGenerationConfigSafety:
    @pytest.mark.parametrize("kwargs", [
        {"model_name": ""},
        {"model_name": "m", "base_url": ""},
        {"model_name": "m", "endpoint": ""},
        {"model_name": "m", "temperature": -1},
        {"model_name": "m", "temperature": 5},
        {"model_name": "m", "max_tokens": 0},
        {"model_name": "m", "top_p": 0.0},
        {"model_name": "m", "top_p": 2.0},
        {"model_name": "m", "repeat_penalty": 0.0},
        {"model_name": "m", "timeout": 0.0},
        {"model_name": "m", "max_retries": -1},
        {"model_name": "m", "retry_delay": -1},
        {"model_name": "m", "max_context_chars": 0},
    ])
    def test_invalid_values_rejected(self, kwargs) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(**kwargs)


class TestIndexingConfigSafety:
    def test_index_mode_enum(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            IndexConfig(index_dir=tmp_path, mode="bogus")

    def test_dense_metric_enum(self) -> None:
        with pytest.raises(ValueError):
            DenseIndexConfig(metric="bogus")

    def test_dense_backend_enum(self) -> None:
        with pytest.raises(ValueError):
            DenseIndexConfig(backend="bogus")

    def test_dense_dimension_positive(self) -> None:
        with pytest.raises(ValueError):
            DenseIndexConfig(dimension=0)

    def test_faiss_index_type_enum(self) -> None:
        with pytest.raises(ValueError):
            FAISSConfig(index_type="bogus")

    def test_sparse_tokenizer_enum(self) -> None:
        with pytest.raises(ValueError):
            SparseIndexConfig(tokenizer="bogus")


class TestRetrievalConfigSafety:
    def test_mode_enum(self) -> None:
        with pytest.raises(ValueError):
            RetrievalConfig(mode="bogus")

    def test_sparse_strategy_enum(self) -> None:
        with pytest.raises(ValueError):
            RetrievalConfig(sparse_strategy="bogus")

    @pytest.mark.parametrize("cfg_cls,kwargs", [
        (DenseRetrievalConfig, {"candidate_k": 0}),
        (BM25Config, {"k1": -1}),
        (BM25Config, {"b": -0.1}),
        (BM25Config, {"b": 1.1}),
        (TFIDFConfig, {"candidate_k": 0}),
        (LearnedSparseConfig, {"min_score": -0.1}),
        (FusionConfig, {"rrf_k": 0}),
        (FusionConfig, {"dense_weight": -0.1}),
        (FusionConfig, {"strategy": "bogus"}),
        (FusionConfig, {"combination": "bogus"}),
        (RerankerConfig, {"lexical_weight": 1.5}),
        (RerankerConfig, {"length_weight": -0.1}),
        (RerankerConfig, {"target_length": 0}),
    ])
    def test_invalid_subconfigs_rejected(self, cfg_cls, kwargs) -> None:
        with pytest.raises(ValueError):
            cfg_cls(**kwargs)

    def test_top_k_positive(self) -> None:
        with pytest.raises(ValueError):
            RetrievalConfig(top_k=0)


class TestImmutability:
    @pytest.mark.parametrize("cfg", [
        EmbeddingConfig(provider="x", model_name="m"),
        GenerationConfig(model_name="m"),
        DenseIndexConfig(),
        SparseIndexConfig(),
        FAISSConfig(),
        BM25Config(),
        TFIDFConfig(),
        FusionConfig(),
        RerankerConfig(),
        LearnedSparseConfig(),
        DenseRetrievalConfig(),
        RetrievalConfig(),
    ])
    def test_configs_are_frozen(self, cfg) -> None:
        with pytest.raises(Exception):
            cfg.model_name = "different"  # type: ignore[misc]
