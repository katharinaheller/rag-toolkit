import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from rag.indexing.base import BaseDenseIndex, BaseSparseIndex
from rag.indexing.config import IndexConfig
from rag.indexing.factory import create_dense_index, create_sparse_index
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.indexing.store import DocumentStore
from rag.indexing.types import DenseIndexEntry, StoredDocument, TokenizedEntry

logger = logging.getLogger(__name__)

_REQUIRED_CHUNK_KEYS = frozenset({"id", "document_id", "text", "metadata"})
_REQUIRED_EMBEDDING_KEYS = frozenset({"id", "chunk_id", "document_id"})


@dataclass
class IndexingResult:
    """Result of an indexing build or load operation.

    document_count is -1 after load() (count unknown without a full scan).
    """

    mode: str
    dense_index: Optional[BaseDenseIndex]
    sparse_index: Optional[BaseSparseIndex]
    document_store: DocumentStore
    document_count: int

    def __repr__(self) -> str:
        return (
            f"IndexingResult("
            f"mode={self.mode!r}, "
            f"document_count={self.document_count}, "
            f"dense={type(self.dense_index).__name__ if self.dense_index else None}, "
            f"sparse={type(self.sparse_index).__name__ if self.sparse_index else None}"
            f")"
        )


class IndexingOrchestrator:
    """Joins EmbeddingVector dicts with ChunkRecord dicts to build indexes.

    Tokenization invariant: text is tokenized once here; the same Tokenizer must
    be used for query tokenization in retrieval.
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._document_store = DocumentStore(config.index_dir / "documents.jsonl")
        self._tokenizer: Optional[Tokenizer] = (
            Tokenizer(config.sparse) if config.mode in {"sparse", "hybrid"} else None
        )

    def build(
        self,
        embeddings: Iterable[Dict[str, Any]],
        chunks: Iterable[Dict[str, Any]],
        persist: bool = False,
    ) -> IndexingResult:
        """Build indexes from embedding and chunk data.

        DocumentStore is always written to disk. persist=True additionally
        flushes dense/sparse backend state.
        """
        chunk_map = _build_chunk_map(chunks)
        stored_docs = list(_build_stored_documents(embeddings, chunk_map))

        if not stored_docs:
            raise ValueError(
                "No documents could be built. "
                "Verify embeddings and chunks are non-empty with matching chunk_ids."
            )

        self._document_store.reset()
        self._document_store.write_many(stored_docs)

        dense_index: Optional[BaseDenseIndex] = None
        sparse_index: Optional[BaseSparseIndex] = None

        if self._config.mode in {"dense", "hybrid"}:
            dense_index = create_dense_index(self._config)
            dense_entries: List[DenseIndexEntry] = [
                {"id": d["id"], "dense_vector": d["dense_vector"]}
                for d in stored_docs
                if d.get("dense_vector") is not None
            ]
            if dense_entries:
                dense_index.add(dense_entries)
            else:
                logger.warning(
                    "Mode '%s': no documents carry a dense_vector. Dense index is empty.",
                    self._config.mode,
                )
            if persist:
                dense_index.persist()

        if self._config.mode in {"sparse", "hybrid"}:
            sparse_index = create_sparse_index(self._config)
            tokenizer: Tokenizer = self._tokenizer  # type: ignore[assignment]
            tokenized: List[TokenizedEntry] = [
                {"id": d["id"], "tokens": tokenizer.tokenize(d["text"])}
                for d in stored_docs
            ]
            sparse_index.add(tokenized)
            if persist:
                sparse_index.persist()

        logger.debug(
            "IndexingOrchestrator.build | mode=%s | docs=%d | persist=%s",
            self._config.mode, len(stored_docs), persist,
        )

        return IndexingResult(
            mode=self._config.mode,
            dense_index=dense_index,
            sparse_index=sparse_index,
            document_store=self._document_store,
            document_count=len(stored_docs),
        )

    def load(self) -> IndexingResult:
        """Load previously persisted index backends from disk."""
        dense_index: Optional[BaseDenseIndex] = None
        sparse_index: Optional[BaseSparseIndex] = None

        if self._config.mode in {"dense", "hybrid"}:
            dense_index = create_dense_index(self._config)
            dense_index.load()

        if self._config.mode in {"sparse", "hybrid"}:
            sparse_index = create_sparse_index(self._config)
            sparse_index.load()

        logger.debug("IndexingOrchestrator.load | mode=%s", self._config.mode)

        return IndexingResult(
            mode=self._config.mode,
            dense_index=dense_index,
            sparse_index=sparse_index,
            document_store=self._document_store,
            document_count=-1,
        )


def _build_chunk_map(chunks: Iterable[Dict[str, Any]]) -> Dict[str, Dict]:
    chunk_map: Dict[str, Dict] = {}
    for chunk in chunks:
        _validate_chunk_record(chunk)
        chunk_map[chunk["id"]] = chunk
    return chunk_map


def _build_stored_documents(
    embeddings: Iterable[Dict[str, Any]],
    chunk_map: Dict[str, Dict],
) -> Iterator[StoredDocument]:
    for emb in embeddings:
        _validate_embedding_record(emb)
        chunk = chunk_map.get(emb["chunk_id"])
        if chunk is None:
            raise ValueError(
                f"Chunk not found for chunk_id='{emb['chunk_id']}' "
                f"(embedding id='{emb['id']}', document_id='{emb['document_id']}')."
            )
        yield {
            "id": emb["id"],
            "chunk_id": chunk["id"],
            "document_id": emb["document_id"],
            "text": chunk["text"],
            "dense_vector": emb.get("embedding"),
            "sparse_vector": emb.get("sparse_embedding"),
            "embedding_type": emb.get("embedding_type", "dense"),
            "metadata": chunk.get("metadata", {}),
        }


def _validate_chunk_record(chunk: Dict[str, Any]) -> None:
    missing = _REQUIRED_CHUNK_KEYS - chunk.keys()
    if missing:
        raise ValueError(
            f"ChunkRecord missing required fields: {sorted(missing)}. "
            f"Present: {sorted(chunk.keys())}."
        )
    if not isinstance(chunk["text"], str) or not chunk["text"].strip():
        raise ValueError(f"ChunkRecord id='{chunk.get('id')}': 'text' must be a non-empty string.")


def _validate_embedding_record(emb: Dict[str, Any]) -> None:
    missing = _REQUIRED_EMBEDDING_KEYS - emb.keys()
    if missing:
        raise ValueError(
            f"EmbeddingVector missing required fields: {sorted(missing)}. "
            f"Present: {sorted(emb.keys())}."
        )
