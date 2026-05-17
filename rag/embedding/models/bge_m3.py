"""BGE-M3 embedder (BAAI/bge-m3).

Loaded via FlagEmbedding. Supports dense, sparse, and hybrid retrieval.
Native dense dimension: 1024. Max sequence length: 8192.
Use config.use_fp16=True (not use_bfloat16) for this model.
"""

from typing import Dict, List, Optional, Set, Tuple

from rag.embedding.base import BaseEmbedder
from rag.embedding.config import EmbeddingConfig
from rag.embedding.model_cache import ModelCache, get_default_cache

try:
    from FlagEmbedding import BGEM3FlagModel
    _FLAG_AVAILABLE = True
except ImportError:
    _FLAG_AVAILABLE = False

_NATIVE_DIMENSION: int = 1024
_NATIVE_MAX_SEQ_LENGTH: int = 8192
_DEFAULT_RETRIEVAL_MODE: str = "hybrid"


class BGEM3Embedder(BaseEmbedder):
    """BGE-M3 multi-function embedder."""

    def __init__(self, config: EmbeddingConfig, cache: Optional[ModelCache] = None) -> None:
        if not _FLAG_AVAILABLE:
            raise ImportError(
                "BGEM3Embedder requires FlagEmbedding. Install: pip install FlagEmbedding"
            )

        self.config = config
        _cache = cache or get_default_cache()

        dtype_label = "fp16" if config.use_fp16 else "float32"
        cache_key = ModelCache.make_key(config.model_name, config.device, dtype_label)

        self.model: BGEM3FlagModel = _cache.get_or_load(
            key=cache_key,
            loader=lambda: self._load_model(config),
        )

    @staticmethod
    def _load_model(config: EmbeddingConfig) -> "BGEM3FlagModel":
        return BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16)

    def _encode(self, texts: List[str], return_dense: bool, return_sparse: bool) -> dict:
        max_length = self.config.max_seq_length or _NATIVE_MAX_SEQ_LENGTH
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

    @staticmethod
    def _sparse_to_dict(lexical_weights: list) -> List[Dict[str, float]]:
        # Convert int token IDs to str for JSON serialization.
        return [{str(k): float(v) for k, v in lw.items()} for lw in lexical_weights]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        output = self._encode(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"].tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        output = self._encode(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"].tolist()

    def embed_documents_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        output = self._encode(texts, return_dense=False, return_sparse=True)
        return self._sparse_to_dict(output["lexical_weights"])

    def embed_queries_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        output = self._encode(texts, return_dense=False, return_sparse=True)
        return self._sparse_to_dict(output["lexical_weights"])

    def embed_documents_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        output = self._encode(texts, return_dense=True, return_sparse=True)
        return output["dense_vecs"].tolist(), self._sparse_to_dict(output["lexical_weights"])

    def embed_queries_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        output = self._encode(texts, return_dense=True, return_sparse=True)
        return output["dense_vecs"].tolist(), self._sparse_to_dict(output["lexical_weights"])

    def dimension(self) -> int:
        return _NATIVE_DIMENSION

    def supported_modes(self) -> Set[str]:
        return {"dense", "sparse", "hybrid"}

    def default_retrieval_mode(self) -> str:
        return _DEFAULT_RETRIEVAL_MODE

    def model_type(self) -> str:
        return "bge-m3"

    def validate_projection_config(
        self,
        target_dim: Optional[int],
        resolved_method: str,
        original_method: str,
        model_aware: bool,
    ) -> None:
        if target_dim is None or resolved_method == "none":
            return

        if resolved_method == "mrl":
            raise ValueError(
                f"BGE-M3 does not support MRL projection. "
                f"(method='{original_method}', model_aware={model_aware}, "
                f"resolved='{resolved_method}'.) "
                "Set model_aware=False and method='truncate' for lossy reduction."
            )

        if target_dim > _NATIVE_DIMENSION:
            raise ValueError(
                f"BGE-M3 native dimension is {_NATIVE_DIMENSION}. "
                f"target_dim={target_dim} exceeds it. Use method='pad' or a smaller target."
            )
