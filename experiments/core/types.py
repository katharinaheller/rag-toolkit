"""Typed records used throughout the experiment framework.

Kept deliberately small. All experiment outputs are written as JSONL/CSV via
``experiments.storage`` so these types only need to round-trip cleanly to
dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


QUERY_TYPES = (
    "keyword",
    "semantic_paraphrase",
    "ambiguous",
    "noisy",
    "multi_hop",
)


@dataclass(frozen=True)
class SyntheticQuery:
    """A synthesised evaluation query with provenance-anchored gold labels.

    ``relevant_chunk_ids`` come from the chunk(s) the query was derived from.
    They are the only reliable proxy for gold relevance in a corpus without
    human labels.
    """

    query_id: str
    text: str
    query_type: str
    corpus: str
    relevant_chunk_ids: List[str]
    relevant_document_ids: List[str]
    source_chunk_excerpt: str = ""
    expected_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalRecord:
    """One retrieval result for one query, stored verbatim."""

    run_id: str
    suite: str
    retriever: str
    embedder: Optional[str]
    corpus: str
    top_k: int
    query_id: str
    query_text: str
    query_type: str
    rank: int
    chunk_id: str
    document_id: str
    score: float
    text_excerpt: str
    latency_ms: float
    repeat_index: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRecord:
    """One generation answer for one (query, retriever) pair."""

    run_id: str
    suite: str
    retriever: str
    embedder: Optional[str]
    corpus: str
    top_k: int
    query_id: str
    query_text: str
    query_type: str
    expected_answer: str
    generated_answer: str
    success: bool
    latency_ms: float
    prompt_chars: int
    context_chars: int
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricRecord:
    """A scalar metric aggregated over many queries."""

    run_id: str
    suite: str
    retriever: str
    embedder: Optional[str]
    corpus: str
    top_k: int
    metric: str
    value: float
    n: int
    query_type: Optional[str] = None
    repeat_index: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
