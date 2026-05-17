"""A single, parameterised callable benchmark invocation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class BenchmarkCase:
    """Wraps a zero-argument callable with a name and optional metadata."""

    name: str
    fn: Callable[[], Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run(self) -> float:
        """Execute fn() and return elapsed wall-clock time in milliseconds."""
        t0 = time.perf_counter()
        self.fn()
        return (time.perf_counter() - t0) * 1_000.0
