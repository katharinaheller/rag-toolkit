# Command Line Interface

This page documents every command-line entry point in the repository so you do
not have to read the source to discover them. There are four Python CLIs (all
built on `argparse`), one standalone report builder, and a set of operator
helpers (a `Makefile`, shell scripts, and a PowerShell script) for the GPU
benchmark stack.

!!! note "Run everything from the repository root"
    ```bash
    cd rag-toolkit
    ```
    All commands below assume the repository root as the working directory.

Every tool also responds to `--help`:

```bash
uv run experiments/run_all_experiments.py --help
uv run analysis/analyse_results.py --help
uv run analysis/plot_latency.py --help
uv run python -m experiments.tools.rescore_generation --help
```

## Overview

| Command                                          | Purpose                                                         |
|--------------------------------------------------|-----------------------------------------------------------------|
| `experiments/run_all_experiments.py`             | Run the full experiment matrix (the main entry point)           |
| `experiments/reports/report_builder.py`          | Rebuild Markdown reports from persisted results (no re-run)     |
| `experiments/tools/rescore_generation.py`        | Re-score stored generation records offline (no Ollama)          |
| `analysis/analyse_results.py`                    | Print thesis-ready comparison tables from stored eval results   |
| `analysis/plot_latency.py`                       | Generate publication-quality figures from stored results        |
| `Makefile` / `benchmark.ps1` / `ops/*.sh`        | Operator helpers for the GPU benchmark stack                    |

!!! warning "Two different result locations"
    The experiment framework (`run_all_experiments.py`) writes to
    `experiments/outputs/`. The analysis tools (`analyse_results.py`,
    `plot_latency.py`) read from `results/eval/` and `results/benchmarks/`,
    which are produced by the `rag.evaluation` JSONL stores (for example from the
    notebooks or your own scripts) — **not** by the experiment framework. If the
    analysis tools report "no results", you are likely pointing them at the wrong
    directory; use their `--eval-dir` / `--benchmark-dir` options.

---

## `run_all_experiments.py`

The primary entry point. Runs the experiment matrix and writes a report,
dashboard, and manifest. See [Running Experiments](experiments.md) for the full
workflow and the description of every suite.

```bash
uv run experiments/run_all_experiments.py [OPTIONS]
```

| Option                  | Type / values            | Default                | Purpose                                                          |
|-------------------------|--------------------------|------------------------|------------------------------------------------------------------|
| `--only-suites`         | comma-separated string   | *(all suites)*         | Suite short ids or full keys to run (`s01,s05` or full key)      |
| `--corpora`             | comma-separated string   | `n10,n50,n100,n1000`   | Corpus names to load                                             |
| `--n-queries`           | int                      | `60`                   | Synthetic queries generated per corpus                           |
| `--seed`                | int                      | `1337`                 | Random seed for deterministic query generation                   |
| `--enable-generation`   | flag                     | off                    | Force-enable the generation suites (`s10`, `s11`, `s17`)         |
| `--disable-generation`  | flag                     | off                    | Force-disable the generation suites                              |
| `--log-level`           | string                   | `INFO`                 | Logging level (`DEBUG`, `INFO`, `WARNING`, …)                    |

**Examples**

```bash
# Retrieval-only full run (no Ollama)
uv run experiments/run_all_experiments.py

# Full run including generation (Ollama + a pulled model required)
uv run experiments/run_all_experiments.py --enable-generation

# Only the three generation suites, on small corpora, for a fast iteration
uv run experiments/run_all_experiments.py \
    --enable-generation \
    --only-suites s10,s11,s17 \
    --corpora n10,n50 \
    --n-queries 10
```

**Exit codes:** `0` success · `1` every suite failed · `2` no corpus loaded.

---

## `report_builder.py`

Rebuilds the per-run `REPORT.md` (for the most recent run) and the cross-run
`AGGREGATE_REPORT.md` from the `_suite_summary.json` files persisted under
`experiments/outputs/aggregated/<run_id>/`. No experiment is re-executed.

```bash
uv run experiments/reports/report_builder.py
```

This tool takes no options. Use it after editing report formatting, or to
regenerate a report that was deleted, without paying the cost of re-running the
suites.

---

## `rescore_generation.py`

An offline re-scoring tool. It reads stored generation records
(`raw/<run_id>/<suite>/generation_records.jsonl`), recomputes the faithfulness
metrics (`context_overlap`, `hallucination_score`) against the stored answers,
and rewrites the aggregated CSVs and figures. It does **not** call Ollama, so no
model is needed.

```bash
uv run python -m experiments.tools.rescore_generation --run-id <RUN_ID> [OPTIONS]
```

| Option           | Type / values | Default                          | Purpose                                                      |
|------------------|---------------|----------------------------------|--------------------------------------------------------------|
| `--run-id`       | string        | *(required)*                     | Run id under `raw/` and `aggregated/`                        |
| `--output-root`  | path          | `EXPERIMENTS_OUTPUT_ROOT`        | Override the output root                                     |
| `--suite`        | string        | *(all rescorable suites)*        | Restrict to a specific suite; repeatable                     |
| `--verbose`      | flag          | off                              | Verbose logging                                              |

The original raw file is backed up to `generation_records.pre_rescore.jsonl`
before being overwritten. See `experiments/CHANGELOG_bugfix.md` for the
background on why this tool exists.

**Example (inside the benchmark-runner container):**

```bash
docker exec -i rag-benchmark-runner \
  python -m experiments.tools.rescore_generation \
    --run-id run_20260527_144558_fbf83b \
    --output-root /opt/experiment-outputs
```

---

## `analyse_results.py`

Generates thesis-ready ASCII comparison tables from stored evaluation and
benchmark JSONL files and writes them to an output directory.

```bash
uv run analysis/analyse_results.py [OPTIONS]
```

| Option            | Type / values | Default               | Purpose                                                      |
|-------------------|---------------|-----------------------|--------------------------------------------------------------|
| `--eval-dir`      | path          | `results/eval`        | Directory of `EvaluationRunResult` JSONL files               |
| `--benchmark-dir` | path          | `results/benchmarks`  | Directory of `BenchmarkResult` JSONL files                   |
| `--output-dir`    | path          | `analysis/reports`    | Where generated table files are written                      |
| `--list-runs`     | flag          | off                   | List available run ids and exit, without generating tables   |

It prints a retrieval comparison table, a generation comparison table, an
end-to-end comparison table, a performance benchmark table, and (if present) a
top-k ablation table.

**Examples**

```bash
# List which runs are available
uv run analysis/analyse_results.py --list-runs

# Generate all tables from a custom results directory
uv run analysis/analyse_results.py --eval-dir my_results/eval
```

---

## `plot_latency.py`

Generates publication-quality figures (latency boxplots, corpus scaling,
concurrency scaling, top-k quality/latency) from the same stored results.

```bash
uv run analysis/plot_latency.py [OPTIONS]
```

| Option            | Type / values            | Default               | Purpose                                                      |
|-------------------|--------------------------|-----------------------|--------------------------------------------------------------|
| `--benchmark-dir` | path                     | `results/benchmarks`  | Directory of benchmark JSONL files                           |
| `--eval-dir`      | path                     | `results/eval`        | Directory of evaluation JSONL files                          |
| `--output-dir`    | path                     | `analysis/figures`    | Where figures are written                                    |
| `--dpi`           | int                      | `150`                 | Figure resolution                                            |
| `--format`        | `png` \| `pdf` \| `svg`  | `png`                 | Output image format                                          |
| `--style`         | string                   | `default`             | Matplotlib style (e.g. `seaborn-v0_8`, `ggplot`, `default`)  |
| `--no-show`       | flag                     | off                   | Do not call `plt.show()` (use in headless / CI environments) |

**Example**

```bash
uv run analysis/plot_latency.py --format pdf --dpi 300 --no-show
```

---

## Operator helpers (GPU benchmark stack)

These wrap the GPU Compose stack and the dedicated `rag-benchmark-runner`
container. They are convenience entry points for the opt-in GPU workflow — see
[GPU Experiments](gpu-experiments.md) for the full picture. They are not
required for local CPU-only experiments.

### Makefile (Linux/macOS)

```bash
make gpu-up      # build images + start the GPU stack incl. benchmark-runner
make hf-verify   # verify the HuggingFace token authenticates
make smoke       # verify the runner sees the GPU
make bench       # run the full GPU experiment + benchmark pipeline
make report      # rebuild REPORT.md + AGGREGATE_REPORT.md only
make shell       # interactive shell inside the runner
make logs        # follow stack logs
make ps          # show stack status
make down        # stop + remove the stack
make gpu-bench   # gpu-up + hf-verify + smoke + bench in one go
```

### benchmark.ps1 (Windows / PowerShell)

The same actions, driven from PowerShell. The default action is `all`.

```powershell
.\benchmark.ps1 up         # build + start the stack incl. benchmark-runner
.\benchmark.ps1 hf-verify  # verify the HuggingFace token authenticates
.\benchmark.ps1 smoke      # verify the runner sees the GPU
.\benchmark.ps1 bench      # run the full GPU experiment + benchmark pipeline
.\benchmark.ps1 report     # rebuild REPORT.md + AGGREGATE_REPORT.md only
.\benchmark.ps1 shell      # interactive shell in the runner
.\benchmark.ps1 logs       # follow logs
.\benchmark.ps1 down       # stop + remove the stack
.\benchmark.ps1 all        # up + hf-verify + smoke + bench (default)
```

### Underlying shell scripts

The `make` targets and PowerShell actions ultimately call these scripts inside
the runner container (`ops/`):

| Script                  | Purpose                                                                  |
|-------------------------|--------------------------------------------------------------------------|
| `ops/hf-verify.sh`      | Confirm the HuggingFace token is present and authenticates against the Hub |
| `ops/gpu-smoke.sh`      | Confirm the runner sees the GPU and the `experiments` package imports    |
| `ops/run-benchmarks.sh` | Run the full experiment pipeline, then rebuild the reports               |
| `ops/build-report.sh`   | Rebuild `REPORT.md` and `AGGREGATE_REPORT.md` from persisted artefacts    |

`ops/hf_check_snippet.py` is not a CLI — it is a set of copy-paste notebook
cells for verifying Hugging Face authentication at runtime.
