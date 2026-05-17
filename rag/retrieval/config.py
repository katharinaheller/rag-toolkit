from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    RetrievalMode = Literal["dense", "sparse_bm25", "sparse_tfidf", "sparse_learned", "hybrid"]
    FusionStrategy = Literal["weighted_sum", "rrf"]
    SparseStrategy = Literal["bm25", "tfidf", "learned"]
    HybridCombination = Literal["union", "intersection"]
else:
    RetrievalMode = str
    FusionStrategy = str
    SparseStrategy = str
    HybridCombination = str

_VALID_MODES = {"dense", "sparse_bm25", "sparse_tfidf", "sparse_learned", "hybrid"}
_VALID_FUSION_STRATEGIES = {"weighted_sum", "rrf"}
_VALID_SPARSE_STRATEGIES = {"bm25", "tfidf", "learned"}
_VALID_HYBRID_COMBINATION = {"union", "intersection"}


@dataclass(frozen=True)
class DenseRetrievalConfig:
    candidate_k: int = 100

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")


@dataclass(frozen=True)
class BM25Config:
    """BM25 hyperparameters. k1 controls TF saturation; b controls length normalization."""
    k1: float = 1.5
    b: float = 0.75
    candidate_k: int = 100

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not (0 <= self.b <= 1):
            raise ValueError("b must be in [0, 1]")
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")


@dataclass(frozen=True)
class TFIDFConfig:
    sublinear_tf: bool = True
    candidate_k: int = 100

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")


@dataclass(frozen=True)
class LearnedSparseConfig:
    candidate_k: int = 100
    min_score: float = 0.0
    use_index_pruning: bool = True

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")


@dataclass(frozen=True)
class FusionConfig:
    """Hybrid score fusion configuration."""
    strategy: FusionStrategy = "weighted_sum"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    combination: HybridCombination = "union"
    normalize_scores: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_FUSION_STRATEGIES:
            raise ValueError(f"Invalid fusion strategy: {self.strategy}")
        if self.combination not in _VALID_HYBRID_COMBINATION:
            raise ValueError(f"Invalid combination: {self.combination}")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("weights must be >= 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    lexical_weight: float = 0.5
    length_weight: float = 0.5
    target_length: int = 512

    def __post_init__(self) -> None:
        if not (0 <= self.lexical_weight <= 1):
            raise ValueError("lexical_weight must be in [0,1]")
        if not (0 <= self.length_weight <= 1):
            raise ValueError("length_weight must be in [0,1]")
        if self.target_length <= 0:
            raise ValueError("target_length must be positive")


@dataclass(frozen=True)
class RetrievalConfig:
    """Top-level retrieval configuration."""
    mode: RetrievalMode = "hybrid"
    sparse_strategy: SparseStrategy = "bm25"
    top_k: int = 10

    dense: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    learned_sparse: LearnedSparseConfig = field(default_factory=LearnedSparseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.sparse_strategy not in _VALID_SPARSE_STRATEGIES:
            raise ValueError(f"Invalid sparse strategy: {self.sparse_strategy}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
