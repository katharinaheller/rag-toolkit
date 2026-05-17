"""Unified timing and resource collection for one pipeline invocation."""

from __future__ import annotations

from typing import List, Optional

from rag.evaluation.monitors.resource_snapshot import ResourceSnapshotCollector
from rag.evaluation.monitors.timer import StageTimer
from rag.evaluation.types import ResourceSnapshot, StageTiming


class PerformanceMonitor:
    """Combines StageTimer with ResourceSnapshotCollector.

    Single-threaded use. Create a new instance per prediction.
    """

    def __init__(
        self,
        capture_resources: bool = False,
        gpu_monitoring: bool = False,
    ) -> None:
        self._capture_resources = capture_resources
        self._timer = StageTimer()
        self._snapshots: List[ResourceSnapshot] = []
        self._collector: Optional[ResourceSnapshotCollector] = (
            ResourceSnapshotCollector(gpu_monitoring=gpu_monitoring)
            if capture_resources
            else None
        )

    def begin(self, stage: str) -> None:
        """Start timing a stage and optionally record a resource snapshot."""
        self._timer.start(stage)
        if self._collector is not None:
            self._snapshots.append(self._collector.collect())

    def end(self, stage: str) -> float:
        """Stop a stage, optionally record a snapshot, return elapsed ms."""
        elapsed = self._timer.stop(stage)
        if self._collector is not None:
            self._snapshots.append(self._collector.collect())
        return elapsed

    def stage_timing(self) -> StageTiming:
        return self._timer.to_stage_timing()

    def snapshots(self) -> List[ResourceSnapshot]:
        return list(self._snapshots)
