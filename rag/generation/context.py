from __future__ import annotations

import logging
from typing import Any, List, Union

logger = logging.getLogger(__name__)

_ContextItem = Union[str, dict]


def sanitize_context(items: List[_ContextItem], *, max_context_chars: int) -> List[str]:
    """Convert raw retrieval results into clean, bounded text chunks.

    Steps: extract text, drop empty/invalid items, strip whitespace,
    deduplicate (order-preserving), truncate to character budget.
    """
    if max_context_chars < 1:
        raise ValueError(f"max_context_chars must be >= 1; got {max_context_chars}.")

    raw_texts: List[str] = []
    for idx, item in enumerate(items):
        text = _extract_text(item, idx)
        if text is not None:
            raw_texts.append(text)

    seen: set = set()
    unique: List[str] = []
    for text in raw_texts:
        stripped = text.strip()
        if not stripped:
            continue
        if stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)

    return _truncate_to_budget(unique, max_context_chars)


def _extract_text(item: Any, idx: int) -> str | None:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        value = item.get("text")
        if isinstance(value, str):
            return value
        logger.warning(
            "Context item at index %d is a dict with invalid 'text' key (got %s); skipped.",
            idx, type(value).__name__,
        )
        return None

    logger.warning(
        "Context item at index %d has unsupported type %s; skipped.", idx, type(item).__name__
    )
    return None


def _truncate_to_budget(chunks: List[str], max_chars: int) -> List[str]:
    if not chunks:
        return []

    result: List[str] = []
    cumulative: int = 0

    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        if i == 0 and chunk_len > max_chars:
            logger.warning(
                "First context chunk (%d chars) exceeds max_context_chars=%d; hard-truncated.",
                chunk_len, max_chars,
            )
            result.append(chunk[:max_chars])
            break

        if cumulative + chunk_len > max_chars:
            logger.debug(
                "Context budget reached at chunk %d/%d (%d chars used of %d).",
                i, len(chunks), cumulative, max_chars,
            )
            break

        result.append(chunk)
        cumulative += chunk_len

    return result


class ContextPreparer:
    """Stateful wrapper around sanitize_context bound to a character budget."""

    def __init__(self, max_context_chars: int) -> None:
        if max_context_chars < 1:
            raise ValueError(f"max_context_chars must be >= 1; got {max_context_chars}.")
        self._max_context_chars = max_context_chars

    @property
    def max_context_chars(self) -> int:
        return self._max_context_chars

    def prepare(self, items: List[_ContextItem]) -> List[str]:
        """Prepare raw context items for prompt assembly."""
        return sanitize_context(items, max_context_chars=self._max_context_chars)
