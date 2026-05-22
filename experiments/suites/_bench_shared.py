"""Shared helpers for the GPU-aware benchmark suites (s16-s21).

Centralises: persisting benchmark outcomes to JSONL + CSV, selecting a sample
of documents/queries from the largest available corpus, and turning outcome
lists into the row dicts that the visualisation layer expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.benchmarks.base import BenchmarkOutcome
from experiments.storage import JsonlWriter, write_csv

logger = logging.getLogger(__name__)


def largest_corpus(
    corpora_chunks: Dict[str, List[Dict]],
) -> Tuple[Optional[str], List[Dict]]:
    """Return ``(name, chunks)`` of the biggest loaded corpus, or (None, [])."""
    best_name: Optional[str] = None
    best_chunks: List[Dict] = []
    for name, chunks in corpora_chunks.items():
        if len(chunks) > len(best_chunks):
            best_name = name
            best_chunks = chunks
    return best_name, best_chunks


def sample_documents(chunks: List[Dict], n: int) -> List[str]:
    """Take up to ``n`` chunk texts (deterministic: corpus order)."""
    texts: List[str] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= n:
            break
    return texts


def persist_outcomes(
    outcomes: Sequence[BenchmarkOutcome],
    raw_path: Path,
    csv_path: Path,
) -> None:
    """Write full outcomes to JSONL and the flat rows to CSV."""
    writer = JsonlWriter(raw_path)
    try:
        for outcome in outcomes:
            writer.write(outcome.to_dict())
    finally:
        writer.close()
    write_csv(csv_path, [o.flat_row() for o in outcomes])


def hardware_findings(outcomes: Sequence[BenchmarkOutcome]) -> List[str]:
    """Produce a one-line description of the hardware the benchmark ran on."""
    findings: List[str] = []
    hw = next((o.hardware for o in outcomes if o.hardware is not None), None)
    if hw is None:
        return findings
    if hw.cuda_available:
        findings.append(
            f"CUDA available: {hw.gpu_count} device(s); primary GPU "
            f"**{hw.primary_gpu_name or 'unknown'}**; "
            f"torch {hw.torch.get('torch_version')}, "
            f"CUDA {hw.torch.get('cuda_version')}."
        )
    else:
        findings.append(
            "CUDA not available — GPU variants recorded as skipped. "
            f"torch={hw.torch.get('torch_version') or 'not installed'}; "
            "running CPU-only benchmarks."
        )
    cpu = hw.cpu or {}
    findings.append(
        f"Host CPU: {cpu.get('logical_cores')} logical cores, "
        f"{cpu.get('total_ram_gb')} GB RAM ({hw.platform_str})."
    )
    return findings


def outcomes_table(outcomes: Sequence[BenchmarkOutcome]) -> List[Dict[str, Any]]:
    """Flat CSV-friendly rows for an arbitrary outcome list."""
    return [o.flat_row() for o in outcomes]
