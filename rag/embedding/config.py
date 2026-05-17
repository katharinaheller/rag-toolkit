from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_VALID_MODES = frozenset({"document", "query"})
_VALID_PROJECTION_METHODS = frozenset({"truncate", "mrl", "pad"})
_VALID_RETRIEVAL_MODES = frozenset({"dense", "sparse", "hybrid"})


@dataclass(frozen=True)
class EmbeddingBehaviorConfig:
    """Controls prefix injection and normalization at inference time."""

    normalize: bool = True
    mode: str = "document"
    query_prefix: Optional[str] = None
    document_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"EmbeddingBehaviorConfig.mode must be one of {sorted(_VALID_MODES)}, "
                f"got '{self.mode}'."
            )


@dataclass(frozen=True)
class EmbeddingProjectionConfig:
    """Optional dimensionality projection after encoding.

    method: "truncate" (lossy prefix), "mrl" (Matryoshka, MRL-capable models only),
    or "pad" (zero-extension to a larger dimension).
    model_aware: if True, the orchestrator selects "mrl" automatically for MRL embedders.
    """

    target_dim: Optional[int] = None
    method: str = "truncate"
    model_aware: bool = True

    def __post_init__(self) -> None:
        if self.method not in _VALID_PROJECTION_METHODS:
            raise ValueError(
                f"EmbeddingProjectionConfig.method must be one of "
                f"{sorted(_VALID_PROJECTION_METHODS)}, got '{self.method}'."
            )
        if self.target_dim is not None and self.target_dim <= 0:
            raise ValueError(
                f"EmbeddingProjectionConfig.target_dim must be positive, "
                f"got {self.target_dim}."
            )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Immutable configuration for a single embedding run.

    use_bfloat16 and use_fp16 are mutually exclusive. Gemma uses bfloat16; BGE-M3 uses fp16.
    retrieval_mode=None lets the embedder pick its default via default_retrieval_mode().
    """

    provider: str
    model_name: str
    model_version: Optional[str] = None

    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: Optional[int] = None
    use_bfloat16: bool = False
    use_fp16: bool = False
    retrieval_mode: Optional[str] = None

    behavior: EmbeddingBehaviorConfig = field(default_factory=EmbeddingBehaviorConfig)
    projection: EmbeddingProjectionConfig = field(default_factory=EmbeddingProjectionConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"EmbeddingConfig.batch_size must be positive, got {self.batch_size}.")
        if self.max_seq_length is not None and self.max_seq_length <= 0:
            raise ValueError(
                f"EmbeddingConfig.max_seq_length must be positive, got {self.max_seq_length}."
            )
        if self.use_bfloat16 and self.use_fp16:
            raise ValueError(
                "use_bfloat16 and use_fp16 are mutually exclusive. Set exactly one."
            )
        if self.retrieval_mode is not None and self.retrieval_mode not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"EmbeddingConfig.retrieval_mode must be one of "
                f"{sorted(_VALID_RETRIEVAL_MODES)} or None, got '{self.retrieval_mode}'."
            )
