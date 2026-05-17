from pathlib import Path

from rag.ingestion.storage.chunk_loader import load_chunks


def verify_chunk_counts(expected: int, path: Path) -> None:
    """Raise ValueError if the number of reloaded chunks differs from expected."""
    loaded = sum(1 for _ in load_chunks(path))
    if loaded != expected:
        raise ValueError(
            f"Chunk count mismatch in {path}: expected {expected}, loaded {loaded} "
            f"(delta={expected - loaded})"
        )


def verify_chunk_ids(expected_ids: set[str], path: Path) -> None:
    """Raise ValueError if reloaded chunk IDs differ from expected."""
    loaded_ids = {chunk["id"] for chunk in load_chunks(path)}
    missing = expected_ids - loaded_ids
    unexpected = loaded_ids - expected_ids

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(
                f"missing {len(missing)} IDs: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            parts.append(
                f"unexpected {len(unexpected)} IDs: "
                f"{sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        raise ValueError(f"Chunk ID mismatch in {path}: {'; '.join(parts)}")
