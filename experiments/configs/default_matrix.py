"""Default experiment matrix.

Every entry is a small, declarative dataclass that the pipeline factory
translates into real ``rag.*`` configuration objects. Keeping the matrix
declarative makes the matrix itself the thing reviewers read first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EmbedderSpec:
    """Declarative description of an embedding configuration."""

    key: str
    provider: str          # "bge-m3" | "gemma"
    model_name: str
    dimension: int
    use_fp16: bool = False
    use_bfloat16: bool = False
    supports_sparse: bool = False
    batch_size: int = 16
    max_seq_length: Optional[int] = 512


@dataclass(frozen=True)
class RetrieverSpec:
    """Declarative description of one retrieval configuration."""

    key: str
    mode: str                       # dense | sparse_bm25 | sparse_tfidf | hybrid
    sparse_strategy: str = "bm25"   # bm25 | tfidf | learned
    needs_dense: bool = False
    needs_sparse_lexical: bool = False
    needs_learned_sparse: bool = False
    embedder_key: Optional[str] = None
    fusion_dense_weight: float = 0.5
    fusion_sparse_weight: float = 0.5


# ── Embedding specs ─────────────────────────────────────────────────────────
EMBEDDERS: List[EmbedderSpec] = [
    EmbedderSpec(
        key="bge",
        provider="bge-m3",
        model_name="BAAI/bge-m3",
        dimension=1024,
        use_fp16=True,
        supports_sparse=True,
        batch_size=8,
        max_seq_length=512,
    ),
    EmbedderSpec(
        key="gemma",
        provider="gemma",
        model_name="google/embeddinggemma-300m",
        dimension=768,
        use_bfloat16=True,
        supports_sparse=False,
        batch_size=16,
        max_seq_length=512,
    ),
]

EMBEDDERS_BY_KEY = {e.key: e for e in EMBEDDERS}


# ── Retriever specs ─────────────────────────────────────────────────────────
RETRIEVERS: List[RetrieverSpec] = [
    RetrieverSpec(
        key="tfidf",
        mode="sparse_tfidf",
        sparse_strategy="tfidf",
        needs_sparse_lexical=True,
    ),
    RetrieverSpec(
        key="bm25",
        mode="sparse_bm25",
        sparse_strategy="bm25",
        needs_sparse_lexical=True,
    ),
    RetrieverSpec(
        key="dense_gemma",
        mode="dense",
        needs_dense=True,
        embedder_key="gemma",
    ),
    RetrieverSpec(
        key="dense_bge",
        mode="dense",
        needs_dense=True,
        embedder_key="bge",
    ),
    RetrieverSpec(
        key="hybrid_gemma",
        mode="hybrid",
        sparse_strategy="bm25",
        needs_dense=True,
        needs_sparse_lexical=True,
        embedder_key="gemma",
    ),
    RetrieverSpec(
        key="hybrid_bge",
        mode="hybrid",
        sparse_strategy="bm25",
        needs_dense=True,
        needs_sparse_lexical=True,
        embedder_key="bge",
    ),
]

RETRIEVERS_BY_KEY = {r.key: r for r in RETRIEVERS}


# ── Corpora and sweeps ──────────────────────────────────────────────────────
CORPORA: List[str] = ["n10", "n50", "n100", "n1000"]

TOPK_VALUES: List[int] = [1, 3, 5, 10, 20]

# Fusion-weight sweep for hybrid retriever analysis.
FUSION_WEIGHT_SWEEP: List[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Stability: how many seeded re-runs of each query.
STABILITY_REPEATS: int = 5

# Benchmark sample size for latency runs.
BENCH_WARMUP: int = 2
BENCH_MEASURED: int = 25

# ── GPU-aware benchmark sweeps (suites s16-s21) ─────────────────────────────
# Batch sizes swept by the embedding benchmark (CPU vs GPU throughput scaling).
BENCH_BATCH_SIZES: List[int] = [1, 4, 8, 16, 32, 64]

# Concurrency levels swept by the retrieval concurrency benchmark.
BENCH_CONCURRENCY_LEVELS: List[int] = [1, 2, 4, 8, 16]

# Number of warm steady-state iterations in the cold-vs-warm benchmark.
COLD_WARM_ITERATIONS: int = 10

# Repeats for the cold-vs-warm benchmark (each repeat rebuilds the model).
COLD_WARM_REPEATS: int = 3


@dataclass(frozen=True)
class Matrix:
    """Convenience grouping for the default sweep."""

    retrievers: List[RetrieverSpec] = field(default_factory=lambda: RETRIEVERS)
    corpora: List[str] = field(default_factory=lambda: CORPORA)
    topk_values: List[int] = field(default_factory=lambda: TOPK_VALUES)


DEFAULT_MATRIX = Matrix()
