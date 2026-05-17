from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    IndexMode = Literal["dense", "sparse", "hybrid"]
    MetricType = Literal["cosine", "dot", "l2"]
    DenseBackend = Literal["brute_force", "faiss"]
    TokenizerType = Literal["simple", "whitespace"]
    FaissIndexType = Literal["flat", "hnsw", "ivf"]
else:
    IndexMode = str
    MetricType = str
    DenseBackend = str
    TokenizerType = str
    FaissIndexType = str

_VALID_MODES = {"dense", "sparse", "hybrid"}
_VALID_METRICS = {"cosine", "dot", "l2"}
_VALID_DENSE_BACKENDS = {"brute_force", "faiss"}
_VALID_TOKENIZERS = {"simple", "whitespace"}
_VALID_FAISS_TYPES = {"flat", "hnsw", "ivf"}


@dataclass(frozen=True)
class FAISSConfig:
    """FAISS-specific indexing parameters."""

    index_type: FaissIndexType = "flat"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    ivf_nlist: int = 100
    search_k_factor: int = 5

    def __post_init__(self) -> None:
        if self.index_type not in _VALID_FAISS_TYPES:
            raise ValueError(f"index_type must be one of {_VALID_FAISS_TYPES}, got '{self.index_type}'")
        if self.search_k_factor < 1:
            raise ValueError("search_k_factor must be >= 1")


@dataclass(frozen=True)
class DenseIndexConfig:
    """Configuration for dense vector indexing."""

    backend: DenseBackend = "brute_force"
    metric: MetricType = "cosine"
    dimension: Optional[int] = None
    faiss: FAISSConfig = field(default_factory=FAISSConfig)

    def __post_init__(self) -> None:
        if self.backend not in _VALID_DENSE_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_DENSE_BACKENDS}, got '{self.backend}'")
        if self.metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {_VALID_METRICS}, got '{self.metric}'")
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("dimension must be positive")


@dataclass(frozen=True)
class SparseIndexConfig:
    """Configuration for sparse lexical indexing. BM25 parameters live in retrieval."""

    tokenizer: TokenizerType = "simple"

    def __post_init__(self) -> None:
        if self.tokenizer not in _VALID_TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {_VALID_TOKENIZERS}, got '{self.tokenizer}'")


@dataclass(frozen=True)
class IndexConfig:
    """Top-level indexing configuration."""

    index_dir: Path
    mode: IndexMode = "dense"
    dense: DenseIndexConfig = field(default_factory=DenseIndexConfig)
    sparse: SparseIndexConfig = field(default_factory=SparseIndexConfig)

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{self.mode}'")
