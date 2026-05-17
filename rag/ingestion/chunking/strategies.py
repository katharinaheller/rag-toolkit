from dataclasses import dataclass, field

from rag.ingestion.ids import ChunkIdFn, positional_chunk_id


@dataclass(frozen=True)
class ChunkingStrategy:
    """Bundles a strategy name with its chunk ID derivation function.

    Use content_chunk_id (instead of the default positional one) for drift
    detection across pipeline runs.
    """
    name: str
    chunk_id_fn: ChunkIdFn = field(default=positional_chunk_id)


FIXED_OVERLAP = ChunkingStrategy("fixed_overlap")
STRUCTURE_AWARE = ChunkingStrategy("structure_aware")
SYNTAX_AWARE = ChunkingStrategy("syntax_aware")
SCHEMA_AWARE = ChunkingStrategy("schema_aware")
