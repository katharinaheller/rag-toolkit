"""Fake retriever and generation strategy implementations for tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeRetriever:
    """Returns a canned list of retrieval results."""

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self._results = results
        self.calls: List[Dict[str, Any]] = []

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        self.calls.append({"query": query, "k": k})
        return list(self._results[: k] if k is not None else self._results)


class RaisingRetriever:
    """Always raises — used to verify error capture in pipelines."""

    def __init__(self, exc: Exception = RuntimeError("retrieval failed")) -> None:
        self._exc = exc

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        raise self._exc
