from typing import Any, Dict

from rag.ingestion.chunking.strategies import ChunkingStrategy


def build_chunk(
    document_id: str,
    index: int,
    text: str,
    base_metadata: Dict[str, Any],
    extra_metadata: Dict[str, Any],
    strategy: ChunkingStrategy,
) -> Dict[str, Any]:
    """Construct a Chunk with a deterministic, collision-resistant ID.

    All chunkers call this so every chunk shares the same structure and ID logic.
    """
    assert document_id, "document_id must not be empty"
    assert text, "chunk text must not be empty"

    return {
        "id": strategy.chunk_id_fn(document_id, index, text, strategy.name),
        "document_id": document_id,
        "text": text,
        "metadata": {
            **base_metadata,
            **extra_metadata,
            "chunk_index": index,
            "strategy": strategy.name,
        },
    }
