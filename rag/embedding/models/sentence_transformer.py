from dataclasses import replace
from typing import List, Optional, Set

from rag.embedding.base import BaseEmbedder
from rag.embedding.config import EmbeddingConfig
from rag.embedding.model_cache import ModelCache, get_default_cache

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class _SentenceTransformerDenseBackend(BaseEmbedder):
    """Internal dense embedding backend via sentence-transformers.

    Not a public provider. Used as a building block for concrete embedders.
    """

    DEFAULT_QUERY_PREFIX: Optional[str] = None
    DEFAULT_DOCUMENT_PREFIX: Optional[str] = None

    def __init__(self, config: EmbeddingConfig, cache: Optional[ModelCache] = None) -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install: pip install sentence-transformers"
            )

        resolved_behavior = replace(
            config.behavior,
            query_prefix=(
                config.behavior.query_prefix
                if config.behavior.query_prefix is not None
                else self.DEFAULT_QUERY_PREFIX
            ),
            document_prefix=(
                config.behavior.document_prefix
                if config.behavior.document_prefix is not None
                else self.DEFAULT_DOCUMENT_PREFIX
            ),
        )
        self.config: EmbeddingConfig = replace(config, behavior=resolved_behavior)

        dtype_label = "bfloat16" if config.use_bfloat16 else "float32"
        cache_key = ModelCache.make_key(config.model_name, config.device, dtype_label)

        _cache = cache or get_default_cache()
        self.model: SentenceTransformer = _cache.get_or_load(
            key=cache_key,
            loader=lambda: self._load_model(config),
        )

    @staticmethod
    def _load_model(config: EmbeddingConfig) -> "SentenceTransformer":
        kwargs = {}
        if config.use_bfloat16:
            try:
                import torch
                kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
            except ImportError:
                raise ImportError("use_bfloat16=True requires torch. Install: pip install torch")

        model = SentenceTransformer(config.model_name, device=config.device, **kwargs)

        if config.max_seq_length is not None:
            model.max_seq_length = config.max_seq_length

        return model

    def _apply_prefix(self, texts: List[str], mode: str) -> List[str]:
        if mode == "query" and self.config.behavior.query_prefix:
            return [f"{self.config.behavior.query_prefix}{t}" for t in texts]
        if mode == "document" and self.config.behavior.document_prefix:
            return [f"{self.config.behavior.document_prefix}{t}" for t in texts]
        return texts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            self._apply_prefix(texts, "document"), convert_to_numpy=True
        ).tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            self._apply_prefix(texts, "query"), convert_to_numpy=True
        ).tolist()

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def supported_modes(self) -> Set[str]:
        return {"dense"}

    def default_retrieval_mode(self) -> str:
        return "dense"

    def model_type(self) -> str:
        return "sentence-transformer-backend"
