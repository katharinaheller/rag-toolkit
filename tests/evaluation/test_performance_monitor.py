"""Tests for the PerformanceMonitor and Timer utilities."""

import time

import pytest

from rag.evaluation.monitors.performance_monitor import PerformanceMonitor
from rag.evaluation.monitors.timer import StageTimer, Timer
from rag.evaluation.benchmarks.benchmark_runner import _compute_stats


class TestTimer:
    def test_elapsed_is_none_before_stop(self):
        t = Timer()
        t.start()
        assert t.elapsed_ms is None  # Not stopped yet.

    def test_elapsed_positive(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        ms = t.stop()
        assert ms > 0.0

    def test_stop_without_start_raises(self):
        with pytest.raises(RuntimeError):
            Timer().stop()

    def test_context_manager(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms is not None
        assert t.elapsed_ms > 0.0


class TestStageTimer:
    def test_timing_multiple_stages(self):
        st = StageTimer()
        st.start("retrieval")
        time.sleep(0.01)
        st.stop("retrieval")
        st.start("generation")
        time.sleep(0.01)
        st.stop("generation")

        assert st.elapsed_ms("retrieval") > 0.0
        assert st.elapsed_ms("generation") > 0.0

    def test_to_stage_timing(self):
        st = StageTimer()
        st.start("end_to_end")
        st.stop("end_to_end")
        timing = st.to_stage_timing()
        assert timing.end_to_end_ms is not None
        assert timing.end_to_end_ms > 0.0
        assert timing.retrieval_ms is None  # Not measured.

    def test_stop_unstated_stage_raises(self):
        st = StageTimer()
        with pytest.raises(KeyError):
            st.stop("nonexistent_stage")


class TestPerformanceMonitor:
    def test_basic_timing(self):
        monitor = PerformanceMonitor(capture_resources=False)
        monitor.begin("end_to_end")
        time.sleep(0.01)
        monitor.end("end_to_end")
        timing = monitor.stage_timing()
        assert timing.end_to_end_ms is not None
        assert timing.end_to_end_ms >= 10.0

    def test_no_snapshots_without_capture(self):
        monitor = PerformanceMonitor(capture_resources=False)
        monitor.begin("end_to_end")
        monitor.end("end_to_end")
        assert monitor.snapshots() == []

    def test_snapshots_with_capture(self):
        monitor = PerformanceMonitor(capture_resources=True, gpu_monitoring=False)
        monitor.begin("retrieval")
        monitor.end("retrieval")
        # Should have 2 snapshots: one on begin, one on end
        snaps = monitor.snapshots()
        assert len(snaps) >= 1


class TestComputeStats:
    def test_basic_stats(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _compute_stats(values)
        assert stats.n == 5
        assert stats.mean == pytest.approx(30.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(50.0)

    def test_p95_single_value(self):
        stats = _compute_stats([100.0])
        assert stats.p95 == pytest.approx(100.0)

    def test_empty_values(self):
        stats = _compute_stats([])
        assert stats.n == 0
        assert stats.mean == 0.0
        assert stats.p95 == 0.0

    def test_p95_correctness(self):
        # With 100 values [1..100], p95 should be ~95
        values = list(range(1, 101))
        stats = _compute_stats([float(v) for v in values])
        assert stats.p95 == pytest.approx(95.0, abs=1.0)

    def test_std_dev(self):
        # All same value → std_dev = 0
        stats = _compute_stats([5.0, 5.0, 5.0])
        assert stats.std_dev == pytest.approx(0.0)
