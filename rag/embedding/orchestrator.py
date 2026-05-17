import logging
from typing import Dict, Iterable, Iterator, List, Optional

from rag.embedding.base import BaseEmbedder, VALID_RETRIEVAL_MODES
from rag.embedding.batching import batch_iter
from rag.embedding.config import EmbeddingConfig
from rag.embedding.normalization import normalize_batch
from rag.embedding.projection import project_batch
from rag.embedding.store import EmbeddingStore
from rag.embedding.types import EmbeddingVector
from rag.embedding.utils import embedding_id

logger = logging.getLogger(__name__)

_REQUIRED_CHUNK_KEYS: frozenset = frozenset({"id", "text", "document_id", "metadata"})


def _validate_chunk(chunk: dict, position: int) -> None:
    if not isinstance(chunk, dict):
        raise ValueError(
            f"Chunk at position {position} must be a dict, got {type(chunk).__name__}."
        )
    missing = _REQUIRED_CHUNK_KEYS - chunk.keys()
    if missing:
        raise ValueError(
            f"Chunk at position {position} missing required fields: "
            f"{sorted(missing)}. Present: {sorted(chunk.keys())}."
        )


def _validate_text(text: object, position: int) -> None:
    if not isinstance(text, str):
        raise ValueError(
            f"Chunk at position {position}: 'text' must be str, got {type(text).__name__}."
        )
    if not text.strip():
        raise ValueError(
            f"Chunk at position {position}: 'text' is empty or whitespace-only."
        )


class EmbeddingOrchestrator:
    """Streaming orchestrator for embedding computation."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        config: EmbeddingConfig,
        store: Optional[EmbeddingStore] = None,
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.store = store

        effective_mode = config.retrieval_mode or embedder.default_retrieval_mode()

        supported = embedder.supported_modes()
        if effective_mode not in supported:
            raise ValueError(
                f"Retrieval mode '{effective_mode}' not supported by "
                f"{embedder.model_type()} (supported: {sorted(supported)}). "
                f"config.retrieval_mode='{config.retrieval_mode}', "
                f"embedder default='{embedder.default_retrieval_mode()}'."
            )

        if effective_mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"Resolved retrieval mode '{effective_mode}' not in "
                f"VALID_RETRIEVAL_MODES {sorted(VALID_RETRIEVAL_MODES)}."
            )

        self._retrieval_mode: str = effective_mode

        if self._retrieval_mode == "sparse" and self.config.projection.target_dim is not None:
            raise ValueError("Projection not applicable for sparse-only retrieval.")

        self._projection_method: str = self._resolve_projection_method()

        self.embedder.validate_projection_config(
            target_dim=self.config.projection.target_dim,
            resolved_method=self._projection_method,
            original_method=self.config.projection.method,
            model_aware=self.config.projection.model_aware,
        )

    def _resolve_projection_method(self) -> str:
        if self.config.projection.target_dim is None:
            return "none"
        if self.config.projection.model_aware and self.embedder.is_mrl_model():
            return "mrl"
        return self.config.projection.method

    def run(self, chunks: Iterable[dict], persist: bool = False) -> List[EmbeddingVector]:
        return list(self.stream(chunks, persist=persist))

    def stream(self, chunks: Iterable[dict], persist: bool = False) -> Iterator[EmbeddingVector]:
        if persist and self.store is None:
            raise ValueError("persist=True requires a store.")
        if persist:
            self.store.reset()

        for batch in batch_iter(chunks, self.config.batch_size):
            batch_results = self._process_batch(batch)
            if persist:
                self.store.write_many(batch_results)
            yield from batch_results

    def _process_batch(self, batch: List[dict]) -> List[EmbeddingVector]:
        for i, chunk in enumerate(batch):
            _validate_chunk(chunk, i)
            _validate_text(chunk["text"], i)

        texts = [chunk["text"] for chunk in batch]
        encode_as_query = self.config.behavior.mode == "query"
        n = len(texts)

        dense_vecs: Optional[List[List[float]]]
        sparse_vecs: Optional[List[Dict[str, float]]]

        if self._retrieval_mode == "dense":
            dense_vecs = (
                self.embedder.embed_queries(texts) if encode_as_query
                else self.embedder.embed_documents(texts)
            )
            sparse_vecs = None

        elif self._retrieval_mode == "sparse":
            sparse_vecs = (
                self.embedder.embed_queries_sparse(texts) if encode_as_query
                else self.embedder.embed_documents_sparse(texts)
            )
            dense_vecs = None

        else:
            dense_vecs, sparse_vecs = (
                self.embedder.embed_queries_hybrid(texts) if encode_as_query
                else self.embedder.embed_documents_hybrid(texts)
            )

        if self._projection_method != "none" and dense_vecs:
            dense_vecs = project_batch(dense_vecs, self.config.projection.target_dim, self._projection_method)

        if self.config.behavior.normalize and dense_vecs:
            dense_vecs = normalize_batch(dense_vecs)

        dense_list = dense_vecs if dense_vecs is not None else [None] * n
        sparse_list = sparse_vecs if sparse_vecs is not None else [None] * n

        return [
            EmbeddingVector(
                id=embedding_id(chunk["id"], self.config.model_name, self.config.model_version),
                chunk_id=chunk["id"],
                document_id=chunk["document_id"],
                embedding=dense,
                sparse_embedding=sparse,
                embedding_type=self._retrieval_mode,
                model_type=self.embedder.model_type(),
                projection_method=self._projection_method,
                metadata=chunk["metadata"],
            )
            for chunk, dense, sparse in zip(batch, dense_list, sparse_list)
        ]
