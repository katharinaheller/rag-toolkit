"""Background-thread resource timeline collector.

Drives :class:`ResourceSnapshotCollector` from the existing ``rag`` toolkit at a
fixed interval, producing a time-aligned series of CPU%, RSS, GPU%, VRAM
measurements. Designed for HPC-style timeline plots (utilisation vs wall
clock).

Usage::

    with ResourceTimeline(interval_s=0.25, gpu_monitoring=True) as tl:
        run_workload()
    samples = tl.samples              # List[ResourceSample]
    tl.write_jsonl(path)              # persist for later replay
    tl.write_csv(path)                # tabular export
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceSample:
    """One time-stamped resource measurement."""

    t_s: float                    # seconds since collector start
    timestamp_iso: str
    cpu_percent: Optional[float]
    memory_rss_mb: Optional[float]
    memory_peak_mb: Optional[float]
    gpu_utilisation_percent: Optional[float]
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    gpu_torch_allocated_mb: Optional[float]
    gpu_torch_peak_mb: Optional[float]
    note: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceTimeline:
    """Polls resource state on a background thread.

    The collector is fail-soft on every axis:

    * if the underlying :class:`ResourceSnapshotCollector` cannot import or
      query anything, samples carry ``None`` values rather than raising;
    * if torch is missing, the GPU torch-allocator fields are ``None``;
    * the timer thread is a daemon so it can never block process shutdown.
    """

    def __init__(
        self,
        interval_s: float = 0.25,
        gpu_monitoring: bool = True,
        gpu_index: int = 0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._interval_s = float(interval_s)
        self._gpu_monitoring = gpu_monitoring
        self._gpu_index = int(gpu_index)
        self._samples: List[ResourceSample] = []
        self._notes: List[str] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_perf: Optional[float] = None
        self._collector = None
        self._init_collector()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def _init_collector(self) -> None:
        try:
            from rag.evaluation.monitors.resource_snapshot import ResourceSnapshotCollector
            self._collector = ResourceSnapshotCollector(
                gpu_monitoring=self._gpu_monitoring,
                gpu_index=self._gpu_index,
            )
        except Exception as exc:
            logger.warning("ResourceSnapshotCollector unavailable: %s", exc)
            self._collector = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._start_perf = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, name="resource-timeline", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "ResourceTimeline":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # ── poll loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._capture_one()
            except Exception as exc:  # never let the thread die
                logger.debug("Resource timeline sample failed: %s", exc)
            # Sleep with cancellation responsiveness.
            self._stop_event.wait(self._interval_s)

    def _capture_one(self) -> None:
        now_perf = time.perf_counter()
        t_s = (now_perf - self._start_perf) if self._start_perf is not None else 0.0

        # CPU / RAM / native GPU via existing collector.
        cpu = rss = peak = None
        gpu_util = gpu_used = gpu_total = None
        if self._collector is not None:
            try:
                snap = self._collector.collect()
                cpu = snap.cpu_percent
                rss = snap.memory_rss_mb
                peak = snap.memory_peak_mb
                gpu_util = snap.gpu_utilisation_percent
                gpu_used = snap.gpu_memory_used_mb
                gpu_total = snap.gpu_memory_total_mb
            except Exception:
                pass

        # Torch-side allocator metrics (independent of NVML).
        torch_alloc = torch_peak = None
        if self._gpu_monitoring:
            try:
                from experiments.core.gpu_hardware import (
                    gpu_allocated_memory_mb, gpu_peak_memory_mb,
                )
                torch_alloc = gpu_allocated_memory_mb(self._gpu_index)
                torch_peak = gpu_peak_memory_mb(self._gpu_index)
            except Exception:
                pass

        note: Optional[str] = None
        with self._lock:
            if self._notes:
                note = self._notes.pop(0)

        sample = ResourceSample(
            t_s=round(t_s, 4),
            timestamp_iso=_utc_iso(),
            cpu_percent=cpu,
            memory_rss_mb=rss,
            memory_peak_mb=peak,
            gpu_utilisation_percent=gpu_util,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            gpu_torch_allocated_mb=torch_alloc,
            gpu_torch_peak_mb=torch_peak,
            note=note,
        )
        with self._lock:
            self._samples.append(sample)

    # ── annotations / public access ────────────────────────────────────────
    def annotate(self, label: str) -> None:
        """Tag the next emitted sample with a free-form label (e.g. 'warmup-end')."""
        with self._lock:
            self._notes.append(label)

    @property
    def samples(self) -> List[ResourceSample]:
        with self._lock:
            return list(self._samples)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate timeline statistics (means, peaks, durations)."""
        samples = self.samples
        if not samples:
            return {
                "n_samples": 0,
                "duration_s": 0.0,
                "interval_s": self._interval_s,
                "gpu_monitoring": self._gpu_monitoring,
            }

        def _series(key: str) -> List[float]:
            out: List[float] = []
            for s in samples:
                v = getattr(s, key)
                if v is not None:
                    out.append(float(v))
            return out

        def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
            if not vals:
                return {"n": 0, "mean": None, "max": None, "min": None}
            return {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 3),
                "max": round(max(vals), 3),
                "min": round(min(vals), 3),
            }

        return {
            "n_samples": len(samples),
            "duration_s": round(samples[-1].t_s, 3),
            "interval_s": self._interval_s,
            "gpu_monitoring": self._gpu_monitoring,
            "cpu_percent": _stats(_series("cpu_percent")),
            "memory_rss_mb": _stats(_series("memory_rss_mb")),
            "gpu_utilisation_percent": _stats(_series("gpu_utilisation_percent")),
            "gpu_memory_used_mb": _stats(_series("gpu_memory_used_mb")),
            "gpu_torch_allocated_mb": _stats(_series("gpu_torch_allocated_mb")),
            "gpu_torch_peak_mb": _stats(_series("gpu_torch_peak_mb")),
        }

    # ── persistence ────────────────────────────────────────────────────────
    def write_jsonl(self, path: Path) -> Path:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        return path

    def write_csv(self, path: Path) -> Path:
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [s.to_dict() for s in self.samples]
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(rows[0].keys())
        # ``extras`` is a dict; csv would break — serialise as JSON.
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                if isinstance(row.get("extras"), dict):
                    import json
                    row["extras"] = json.dumps(row["extras"], ensure_ascii=False)
                writer.writerow(row)
        return path


def _utc_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
