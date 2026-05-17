"""Proxy faithfulness and hallucination signals.

Without an external judge LLM, we use context-grounded proxies:

* ``context_overlap``    fraction of answer tokens present in the provided context
* ``hallucination_rate`` 1 − context_overlap, optionally weighted by content-only tokens
* ``answer_token_f1``    token-F1 against expected answer (when available)
* ``answer_em``          exact match against expected answer (when available)

These signals are interpretable and reproducible — preferred over LLM judges
in a local, CPU-bound setup.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple

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


def token_f1(pred: str, gold: str) -> float:
    """Word-overlap F1 (multiset)."""
    p = _tokenise(pred)
    g = _tokenise(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc = Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    precision = common / sum(pc.values())
    recall = common / sum(gc.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower().strip())
    return 1.0 if norm(pred) == norm(gold) and norm(pred) else 0.0


def context_pollution_ratio(
    retrieved_chunk_ids: List[str], gold_chunk_ids: Iterable[str]
) -> float:
    """Fraction of retrieved chunks that are NOT in the gold set."""
    if not retrieved_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    polluted = sum(1 for c in retrieved_chunk_ids if c not in gold)
    return polluted / len(retrieved_chunk_ids)
