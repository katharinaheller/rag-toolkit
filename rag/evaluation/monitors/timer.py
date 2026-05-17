"""Wall-clock timers based on time.perf_counter()."""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """High-resolution single-shot timer.

    Usage::

        timer = Timer()
        timer.start()
        do_work()
        elapsed_ms = timer.stop()

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start().")
        self._end = time.perf_counter()
        return self.elapsed_ms  # type: ignore[return-value]

    @property
    def elapsed_ms(self) -> Optional[float]:
        """Elapsed milliseconds, or None if not yet stopped."""
        if self._start is None or self._end is None:
            return None
        return (self._end - self._start) * 1_000.0

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


class StageTimer:
    """Records per-stage timings for one pipeline invocation."""

    KNOWN_STAGES = frozenset({
        "retrieval", "reranking", "prompt_construction", "generation", "end_to_end"
    })

    def __init__(self) -> None:
        self._timers: dict[str, Timer] = {}

    def start(self, stage: str) -> None:
        t = Timer()
        t.start()
        self._timers[stage] = t

    def stop(self, stage: str) -> float:
        if stage not in self._timers:
            raise KeyError(f"Stage '{stage}' was not started.")
        return self._timers[stage].stop()

    def elapsed_ms(self, stage: str) -> Optional[float]:
        t = self._timers.get(stage)
        return t.elapsed_ms if t is not None else None

    def to_stage_timing(self):
        """Convert recorded stages to a StageTiming dataclass."""
        from rag.evaluation.types import StageTiming
        return StageTiming(
            retrieval_ms=self.elapsed_ms("retrieval"),
            reranking_ms=self.elapsed_ms("reranking"),
            prompt_construction_ms=self.elapsed_ms("prompt_construction"),
            generation_ms=self.elapsed_ms("generation"),
            end_to_end_ms=self.elapsed_ms("end_to_end"),
        )
