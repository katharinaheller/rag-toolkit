"""Assertion helpers that encode invariants the production code must uphold.

Use these in place of bespoke per-test assertions to keep error messages
consistent and to make invariant violations obvious at the call site.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping


def assert_sorted_desc_by_score_then_id(items: List[Mapping[str, Any]]) -> None:
    """Confirm a list is sorted by score DESC, id ASC.

    The (-score, id) ordering is the documented deterministic ordering used
    across every retrieval and indexing component.
    """
    keys = [(-float(it["score"]), str(it["id"])) for it in items]
    assert keys == sorted(keys), f"Items not sorted by (-score, id): {keys}"


def assert_unit_norm(vector: List[float], tol: float = 1e-6) -> None:
    """Assert a vector is L2-normalized to within numerical tolerance."""
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) <= tol, f"Vector L2 norm is {norm}, expected ~1.0"


def assert_all_finite(vector: Iterable[float]) -> None:
    """Assert every component is finite (no NaN/Inf)."""
    for x in vector:
        assert math.isfinite(x), f"Non-finite value encountered: {x}"


def assert_jsonl_lines(path, n: int) -> None:
    """Assert that a JSONL file contains exactly n non-empty lines."""
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    assert len(lines) == n, f"Expected {n} lines, found {len(lines)}"


def assert_keys_present(obj: Mapping[str, Any], required: Iterable[str]) -> None:
    """Assert that every required key exists in obj."""
    missing = [k for k in required if k not in obj]
    assert not missing, f"Missing required keys: {missing}"


def assert_deterministic(fn, *args, runs: int = 3, **kwargs) -> None:
    """Run fn three times with the same args and assert identical output."""
    first = fn(*args, **kwargs)
    for _ in range(runs - 1):
        again = fn(*args, **kwargs)
        assert again == first, "Function output is non-deterministic across calls."
