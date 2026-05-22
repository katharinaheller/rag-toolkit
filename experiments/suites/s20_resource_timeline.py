"""Suite 20 — HPC resource timeline.

Drives the embedding workload on each device under a high-resolution resource
timeline collector and renders the classic HPC utilisation curves (CPU%, GPU%,
RSS, VRAM vs wall clock) plus a GPU utilisation heatmap across batch sizes.

This suite is about *visualising* resource behaviour rather than producing
headline latency numbers, so it always captures the timeline regardless of
CUDA availability (CPU% and RSS curves are still informative on a CPU host).
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.benchmarks import EmbeddingBenchmark
from experiments.benchmarks.base import base_devices
from experiments.configs.default_matrix import EMBEDDERS
from experiments.configs.settings import SETTINGS
from experiments.storage import read_jsonl
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
    sample_documents,
)
from experiments.suites.base import Suite
from experiments.visualisation.resource_plots import (
    plot_gpu_utilisation_heatmap,
    plot_resource_timeline,
)


class ResourceTimelineSuite(Suite):
    key = "s20_resource_timeline"
    description = "HPC-style CPU/GPU/RAM/VRAM utilisation timelines and heatmaps"

    _HEATMAP_BATCHES = [4, 16, 64]

    def run(self) -> Dict[str, Any]:
        if not SETTINGS.enable_gpu_benchmarks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["GPU benchmarks disabled via EXPERIMENTS_ENABLE_GPU_BENCHMARKS=0."],
            }

        corpus_name, chunks = largest_corpus(self.ctx.corpora_chunks)
        if not chunks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["No corpus available for resource timeline."],
            }
        docs = sample_documents(chunks, max(SETTINGS.benchmark_doc_sample, 64))
        spec = EMBEDDERS[0]
        timeline_dir = self.raw_dir / "timelines"

        bench = EmbeddingBenchmark(
            embedder_spec=spec,
            documents=docs,
            batch_sizes=self._HEATMAP_BATCHES,
            warmup_iterations=SETTINGS.benchmark_warmup,
            measured_iterations=max(SETTINGS.benchmark_measured, 5),
            devices=base_devices(SETTINGS.gpu_index),
            seed=SETTINGS.seed,
            resource_interval_s=SETTINGS.resource_interval_s,
            timeline_dir=timeline_dir,
        )
        outcomes = bench.run()
        persist_outcomes(
            outcomes,
            self.raw_path("timeline_outcomes.jsonl"),
            self.agg_path("resource_timeline_summary.csv"),
        )

        figures: List[str] = []
        # One detailed timeline per device at the largest batch size.
        per_device_best: Dict[str, Any] = {}
        gpu_timelines: Dict[str, List[Dict]] = {}
        for o in outcomes:
            if not o.success or not o.resource_timeline_path:
                continue
            from pathlib import Path as _Path
            samples = read_jsonl(_Path(o.resource_timeline_path))
            bs = o.variant.parameters.get("batch_size")
            device = o.variant.device.key
            # Keep the largest batch per device for the detailed timeline.
            prev = per_device_best.get(device)
            if prev is None or bs > prev[0]:
                per_device_best[device] = (bs, samples, o.variant.device.label)
            # Collect all (device, batch) timelines for the heatmap.
            gpu_timelines[f"{device} bs={bs}"] = samples

        for device, (bs, samples, label) in per_device_best.items():
            safe = device.replace(":", "-")
            tl_path = self.figure_path(f"timeline_{safe}.png")
            if plot_resource_timeline(
                samples, tl_path,
                title=f"Resource timeline — {label} (batch={bs})",
            ):
                figures.append(str(tl_path))

        heatmap_path = self.figure_path("gpu_utilisation_heatmap.png")
        if plot_gpu_utilisation_heatmap(
            gpu_timelines, heatmap_path,
            title="GPU utilisation heatmap across batch sizes",
        ):
            figures.append(str(heatmap_path))

        findings = self._interpret(outcomes)
        findings.extend(hardware_findings(outcomes))
        return {
            "figures": figures,
            "tables": [str(self.agg_path("resource_timeline_summary.csv"))],
            "findings": findings,
            "rows": [o.flat_row() for o in outcomes if o.success],
        }

    def _interpret(self, outcomes: List[Any]) -> List[str]:
        findings: List[str] = []
        any_gpu = False
        for o in outcomes:
            if not o.success:
                continue
            tl = o.resource_timeline_summary or {}
            gpu = tl.get("gpu_utilisation_percent") or {}
            cpu = tl.get("cpu_percent") or {}
            device = o.variant.device.key
            bs = o.variant.parameters.get("batch_size")
            if gpu.get("max"):
                any_gpu = True
                findings.append(
                    f"**{device}** batch={bs}: peak GPU utilisation "
                    f"{gpu.get('max'):.0f}%, mean {gpu.get('mean'):.0f}%."
                )
            elif cpu.get("max"):
                findings.append(
                    f"**{device}** batch={bs}: peak CPU utilisation "
                    f"{cpu.get('max'):.0f}%, mean {cpu.get('mean'):.0f}%."
                )
        if not any_gpu:
            findings.append(
                "No GPU utilisation recorded (CUDA unavailable); timelines "
                "show CPU and RSS curves only."
            )
        return findings
