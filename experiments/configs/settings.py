"""Environment-wide settings, paths and default flags.

All paths are resolved through ``Path`` so the package works on Windows,
Linux and inside Docker/JupyterHub. Override anything via the corresponding
``EXPERIMENTS_*`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


# Repository root is the parent of this package's parent.
_PKG_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_DIR.parent


@dataclass(frozen=True)
class Settings:
    """Process-wide configuration for the experiment framework."""

    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)
    data_root: Path = field(default_factory=lambda: _env_path(
        "EXPERIMENTS_DATA_ROOT", _REPO_ROOT / "data" / "documents"
    ))
    output_root: Path = field(default_factory=lambda: _env_path(
        "EXPERIMENTS_OUTPUT_ROOT", _PKG_DIR / "outputs"
    ))
    cache_root: Path = field(default_factory=lambda: _env_path(
        "EXPERIMENTS_CACHE_ROOT", _PKG_DIR / "outputs" / "indexes"
    ))

    # Generation toggle: when False, generation-dependent suites still run
    # but emit a clear "skipped: ollama disabled" record.
    enable_generation: bool = field(default_factory=lambda: _env_bool(
        "EXPERIMENTS_ENABLE_GENERATION", False
    ))
    ollama_base_url: str = field(default_factory=lambda: os.environ.get(
        "EXPERIMENTS_OLLAMA_URL", "http://localhost:11434"
    ))
    ollama_model: str = field(default_factory=lambda: os.environ.get(
        "EXPERIMENTS_OLLAMA_MODEL", "mistral"
    ))
    ollama_timeout_s: float = field(default_factory=lambda: float(
        os.environ.get("EXPERIMENTS_OLLAMA_TIMEOUT", "120.0")
    ))

    # How many synthetic queries to derive per corpus.
    n_queries_per_corpus: int = field(default_factory=lambda: _env_int(
        "EXPERIMENTS_N_QUERIES", 60
    ))

    # Reproducibility.
    seed: int = field(default_factory=lambda: _env_int("EXPERIMENTS_SEED", 1337))

    # Run identifier.
    run_id: str = ""

    # Which suites to execute. Empty list = all.
    only_suites: List[str] = field(default_factory=list)

    # Logging.
    log_level: str = field(default_factory=lambda: os.environ.get(
        "EXPERIMENTS_LOG_LEVEL", "INFO"
    ))

    # ── GPU / benchmark settings ────────────────────────────────────────────
    # Master switch for the GPU-aware benchmark suites (s16-s21). When True the
    # suites still run on CPU-only hosts; the GPU variants merely record a
    # "skipped: CUDA unavailable" outcome. Default True so the benchmarks are
    # part of the standard pipeline.
    enable_gpu_benchmarks: bool = field(default_factory=lambda: _env_bool(
        "EXPERIMENTS_ENABLE_GPU_BENCHMARKS", True
    ))

    # GPU device index used by the resource monitor and benchmarks.
    gpu_index: int = field(default_factory=lambda: _env_int(
        "EXPERIMENTS_GPU_INDEX", 0
    ))

    # Resource timeline sampling interval (seconds).
    resource_interval_s: float = field(default_factory=lambda: _env_float(
        "EXPERIMENTS_RESOURCE_INTERVAL", 0.25
    ))

    # Benchmark iteration counts (kept small so the full pipeline finishes
    # quickly on a CPU-only laptop; override via env for serious runs).
    benchmark_warmup: int = field(default_factory=lambda: _env_int(
        "EXPERIMENTS_BENCH_WARMUP", 1
    ))
    benchmark_measured: int = field(default_factory=lambda: _env_int(
        "EXPERIMENTS_BENCH_MEASURED", 3
    ))

    # How many corpus documents to feed embedding benchmarks (capped per corpus).
    benchmark_doc_sample: int = field(default_factory=lambda: _env_int(
        "EXPERIMENTS_BENCH_DOC_SAMPLE", 64
    ))


SETTINGS = Settings()


def corpus_dir(name: str) -> Path:
    """Resolve a corpus name (e.g. ``n100``) to an absolute path."""
    return SETTINGS.data_root / name


def output_subdir(*parts: str) -> Path:
    path = SETTINGS.output_root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
