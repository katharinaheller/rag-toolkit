"""Proxy faithfulness and hallucination signals.

Without an external judge LLM, we use context-grounded proxies:

* ``context_overlap``         fraction of answer tokens present in the context
* ``hallucination_score``     1 − context_overlap
* ``context_pollution_ratio`` fraction of retrieved chunks not in the gold set

These signals are interpretable and reproducible — preferred over LLM judges
in a local, CPU-bound setup.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]+|\d+")

_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "as", "at", "by", "this", "that",
    "it", "its", "from", "into", "but", "not", "no", "do", "does", "did",
    "has", "have", "had", "will", "would", "can", "could", "may", "might",
    "you", "we", "they", "i", "me", "us", "them",
})


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokenise(text) if t not in _STOP and len(t) > 2]


def context_overlap(answer: str, contexts: Iterable[str]) -> float:
    """Fraction of content tokens in the answer that appear in any context."""
    ans = _content_tokens(answer)
    if not ans:
        return 0.0
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(_content_tokens(c))
    if not ctx_tokens:
        return 0.0
    overlap = sum(1 for t in ans if t in ctx_tokens)
    return overlap / len(ans)


def hallucination_score(answer: str, contexts: Iterable[str]) -> float:
    return 1.0 - context_overlap(answer, contexts)


def context_pollution_ratio(
    retrieved_chunk_ids: List[str], gold_chunk_ids: Iterable[str]
) -> float:
    """Fraction of retrieved chunks that are NOT in the gold set."""
    if not retrieved_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    polluted = sum(1 for c in retrieved_chunk_ids if c not in gold)
    return polluted / len(retrieved_chunk_ids)
