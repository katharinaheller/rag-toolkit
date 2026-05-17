"""EmbeddingGemma embedder (google/embeddinggemma-300m).

Loaded via sentence-transformers. Dense-only retrieval.
Native dimension: 768. Supports MRL at dimensions 512, 256, 128.
Use config.use_bfloat16=True (not use_fp16) — Gemma overflows in float16.

Default query format:    "task: {task_name} | query: {text}"
Default document format: "title: {title} | text: {text}"
"""

from typing import FrozenSet, List, Optional, Set

from rag.embedding.base import BaseEmbedder
from rag.embedding.config import EmbeddingConfig
from rag.embedding.model_cache import ModelCache, get_default_cache

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GemmaEmbedder(BaseEmbedder):
    """EmbeddingGemma embedder."""

    _DEFAULT_QUERY_TEMPLATE: str = "task: search result | query: {text}"
    _DEFAULT_DOCUMENT_TEMPLATE: str = "title:  | text: {text}"
    MRL_VALID_DIMS: FrozenSet[int] = frozenset({768, 512, 256, 128})

    def __init__(self, config: EmbeddingConfig, cache: Optional[ModelCache] = None) -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "GemmaEmbedder requires sentence-transformers. "
                "Install: pip install sentence-transformers"
            )

        self.config = config
        _cache = cache or get_default_cache()

        dtype_label = "bfloat16" if config.use_bfloat16 else "float32"
        cache_key = ModelCache.make_key(config.model_name, config.device, dtype_label)

        self.model: SentenceTransformer = _cache.get_or_load(
            key=cache_key,
            loader=lambda: self._load_model(config),
        )

    @staticmethod
    def _load_model(config: EmbeddingConfig) -> "SentenceTransformer":
        kwargs = {}
        if config.use_bfloat16:
            if not _TORCH_AVAILABLE:
                raise ImportError("use_bfloat16=True requires torch. Install: pip install torch")
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}

        model = SentenceTransformer(config.model_name, device=config.device, **kwargs)

        if config.max_seq_length is not None:
            model.max_seq_length = config.max_seq_length

        return model

    def validate_projection_config(
        self,
        target_dim: Optional[int],
        resolved_method: str,
        original_method: str,
        model_aware: bool,
    ) -> None:
        if target_dim is None or resolved_method != "mrl":
            return
        if target_dim not in self.MRL_VALID_DIMS:
            raise ValueError(
                f"EmbeddingGemma MRL requires target_dim in "
                f"{sorted(self.MRL_VALID_DIMS)}, got {target_dim}. "
                f"(method='{original_method}', model_aware={model_aware}, "
                f"resolved='{resolved_method}'.) "
                "For arbitrary truncation, set model_aware=False and method='truncate'."
            )

    def _format_query(self, text: str) -> str:
        if self.config.behavior.query_prefix:
            return f"{self.config.behavior.query_prefix}{text}"
        return self._DEFAULT_QUERY_TEMPLATE.format(text=text)

    def _format_document(self, text: str) -> str:
        if self.config.behavior.document_prefix:
            return f"{self.config.behavior.document_prefix}{text}"
        return self._DEFAULT_DOCUMENT_TEMPLATE.format(text=text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        formatted = [self._format_document(t) for t in texts]
        return self.model.encode(formatted, convert_to_numpy=True).tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        formatted = [self._format_query(t) for t in texts]
        return self.model.encode(formatted, convert_to_numpy=True).tolist()

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def is_mrl_model(self) -> bool:
        return True

    def supported_modes(self) -> Set[str]:
        return {"dense"}

    def default_retrieval_mode(self) -> str:
        return "dense"

    def model_type(self) -> str:
        return "gemma"
