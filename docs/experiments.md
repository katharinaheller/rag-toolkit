# Running Experiments

The `experiments/` package is a self-contained research framework that runs a
matrix of retrieval, generation, scaling, and resource benchmarks against the
`rag` library and writes Markdown reports, figures, and machine-readable
artefacts.

This page is written for someone who has just cloned the repository and wants
to reproduce the results from a fresh machine. It assumes no prior knowledge of
the internal implementation.

!!! note "Run everything from the repository root"
    Every command on this page is executed from the repository root (the folder
    that contains `pyproject.toml`):

    ```bash
    cd rag-toolkit
    ```

## Prerequisites

Install the project with [uv](https://docs.astral.sh/uv/) first:

```bash
uv sync
```

That installs the `rag` library plus everything the experiment framework needs.
Whether you need anything *else* depends on which suites you run:

| You want to run…                                  | You additionally need                                              |
|---------------------------------------------------|--------------------------------------------------------------------|
| Sparse retrieval suites only (BM25, TF-IDF)       | Nothing — pure CPU, no model downloads                             |
| Dense / hybrid retrieval suites                   | Hugging Face access to download embedding models (see below)       |
| Generation suites (`s10`, `s11`, `s17`)           | A running [Ollama](https://ollama.com/) instance **and** a model   |
| GPU benchmark variants (`s16`–`s21`)              | An NVIDIA GPU (the suites still run CPU-only and mark a clean skip) |

### Embedding model downloads (retrieval already needs these)

The default matrix uses two embedding models. They are downloaded from Hugging
Face on first use and then cached:

- `BAAI/bge-m3` — openly available, no token required (≈ 2 GB).
- `google/embeddinggemma-300m` — **gated**; you must accept its license on
  Hugging Face and authenticate before it can be downloaded (≈ 1 GB).

To authenticate for the gated model, export a read token before running the
experiments (create one at <https://huggingface.co/settings/tokens>):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # Linux/macOS
```

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Windows PowerShell
```

Even a *retrieval-only* run downloads these models, because the dense and hybrid
retrievers depend on them. Only the pure sparse suites (BM25, TF-IDF) avoid
model downloads entirely.

## The two operating modes

The framework has exactly two operating modes. The difference is whether the
**generation** (LLM) suites run.

### Retrieval mode (default — no Ollama)

This is the default. It computes everything that does **not** require a language
model: retrieval quality, scaling, latency, embedding comparisons, robustness,
stability, and the resource benchmarks.

```bash
uv run experiments/run_all_experiments.py
```

What it does:

- Loads each corpus's chunks (via the ingestion pipeline, cached on disk).
- Generates deterministic synthetic queries per corpus.
- Runs every suite **except** the generation-dependent ones — `s10`, `s11`, and
  `s17` detect that no generator is available and record a clean
  `skipped` status instead of failing.
- Emits a dashboard image, a Markdown report, and a run manifest.

Suites that run in retrieval mode: `s01`–`s09`, `s12`–`s16`, `s18`–`s21`.

### Generation mode (needs Ollama + a model)

Generation suites need a local LLM served by Ollama. Two things must be true:

1. **Ollama must be reachable.** By default the framework looks for it at
   `http://localhost:11434`.
2. **The model must be installed.** Ollama being reachable does *not* mean the
   model is present — you must pull it once.

Pull the model (default model name is `mistral`):

```bash
ollama pull mistral
```

Then enable generation explicitly:

```bash
uv run experiments/run_all_experiments.py --enable-generation
```

With generation enabled, the three generation suites become active:

| Suite | What it measures                                                       |
|-------|------------------------------------------------------------------------|
| `s10` | Faithfulness vs. controlled context-pollution ratio                    |
| `s11` | Generation quality and latency as context length grows                 |
| `s17` | Generation latency, throughput, and resource timeline (CPU/GPU view)   |

If `--enable-generation` is passed but Ollama is unreachable or the model is
missing, these suites record a `skipped` status with the reason — the rest of
the run still completes.

## Running selected suites

Running the whole matrix is unnecessary while iterating. Restrict the run with
`--only-suites`, which accepts comma-separated short ids (`s05`) or full keys
(`s05_latency_quality_pareto`):

```bash
uv run experiments/run_all_experiments.py \
    --enable-generation \
    --only-suites s10,s11,s17
```

Use this to:

- **Iterate quickly** on one or two suites without paying for the full matrix.
- **Reproduce a single figure** that appears in the report.
- **Re-run only the generation suites** after starting Ollama, without recomputing
  the retrieval suites.

Selection is deterministic: the same `--seed` (default `1337`) and the same
`--only-suites`, `--corpora`, and `--n-queries` produce the same queries and the
same comparisons on every machine.

## The 21 suites

| Key   | Suite                          | Mode       | What it answers                                                      |
|-------|--------------------------------|------------|----------------------------------------------------------------------|
| `s01` | Retriever × corpus scaling     | retrieval  | How retrieval quality scales from `n10` to `n1000` per retriever     |
| `s02` | Top-k sensitivity              | retrieval  | How recall, precision, and pollution evolve as `top_k` grows         |
| `s03` | Query-type comparison          | retrieval  | Dense vs. sparse vs. hybrid stratified by query type                 |
| `s04` | Embedding comparison           | retrieval  | Head-to-head: EmbeddingGemma vs. BGE-M3                              |
| `s05` | Latency–quality Pareto         | retrieval  | Latency vs. quality trade-offs and the Pareto frontier               |
| `s06` | Retrieval overlap              | retrieval  | Pairwise overlap to find complementary retrievers                    |
| `s07` | Hybrid fusion sweep            | retrieval  | Sweep fusion weights to find the dense/sparse balance                |
| `s08` | Noise robustness               | retrieval  | Retrieval degradation under noisy vs. paraphrased queries            |
| `s09` | Stability                      | retrieval  | Variance and rank stability across repeated retrieval runs           |
| `s10` | Context pollution              | generation | Faithfulness vs. controlled context-pollution ratio                  |
| `s11` | Long context                   | generation | Generation quality and latency as context length grows               |
| `s12` | Failure taxonomy               | retrieval  | Distribution of retrieval failure modes per retriever                |
| `s13` | Chunk-relevance decay          | retrieval  | Per-rank hit rate — where retrieval signal concentrates              |
| `s14` | Throughput & resources         | retrieval  | Latency, throughput, and memory for each retriever × corpus          |
| `s15` | Query-conditioned analysis     | retrieval  | Best retriever per query type and the oracle-routing upper bound     |
| `s16` | CPU vs. GPU embedding          | retrieval  | Embedding latency, throughput, VRAM, and cold start                  |
| `s17` | CPU vs. GPU generation         | generation | Generation latency, throughput, and resource timeline                |
| `s18` | Batch-size scaling             | retrieval  | Embedding throughput and VRAM as a function of batch size            |
| `s19` | Cold vs. warm                  | retrieval  | Cold start (load + first call) vs. warm steady-state latency         |
| `s20` | Resource timeline              | retrieval  | CPU/GPU/RAM/VRAM utilisation timelines and heatmaps                  |
| `s21` | Concurrency scaling            | retrieval  | Retrieval throughput and p95 latency vs. concurrency level           |

"Mode = retrieval" suites run without Ollama. "Mode = generation" suites need
`--enable-generation` plus a reachable model. The GPU-aware suites (`s16`–`s21`)
run on CPU-only hosts and mark the GPU half as skipped when no GPU is present.

## Command-line options

`run_all_experiments.py` exposes the following options. See the
[Command Line Interface](cli.md) page for the complete reference of every CLI in
the repository.

| Option                  | Default                  | Purpose                                                          |
|-------------------------|--------------------------|------------------------------------------------------------------|
| `--only-suites`         | *(all)*                  | Comma-separated suite ids/keys to run (`s01,s05` or full key)    |
| `--corpora`             | `n10,n50,n100,n1000`     | Comma-separated corpus names to load                             |
| `--n-queries`           | `60`                     | Synthetic queries generated per corpus                           |
| `--seed`                | `1337`                   | Random seed for deterministic query generation                   |
| `--enable-generation`   | off                      | Force-enable the generation suites                               |
| `--disable-generation`  | off                      | Force-disable the generation suites                              |
| `--log-level`           | `INFO`                   | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, …)                |

All defaults can also be set through `EXPERIMENTS_*` environment variables (for
example `EXPERIMENTS_N_QUERIES`, `EXPERIMENTS_SEED`,
`EXPERIMENTS_ENABLE_GENERATION`, `EXPERIMENTS_OLLAMA_URL`,
`EXPERIMENTS_OLLAMA_MODEL`). CLI flags always take precedence over the
environment.

## Where the results go

All artefacts are written under a single output root. For a local run that root
is `experiments/outputs/`; inside the Docker benchmark-runner it is
`/opt/experiment-outputs` (set via `EXPERIMENTS_OUTPUT_ROOT`). Every run is
namespaced by a unique `run_id` (for example `run_20260527_144558_fbf83b`).

```
experiments/outputs/
├── logs/<run_id>/
│   ├── run.log                     # full run log
│   └── manifest.json               # machine-readable run manifest
├── raw/<run_id>/<suite>/           # raw per-suite records (JSONL/CSV)
├── aggregated/<run_id>/<suite>/
│   └── _suite_summary.json         # per-suite summary (used to rebuild reports)
├── figures/<run_id>/<suite>/       # per-suite figures (PNG)
├── reports/<run_id>/
│   ├── REPORT.md                   # the per-run Markdown report
│   └── dashboard.png               # four-panel summary dashboard
├── reports/_aggregate/
│   ├── AGGREGATE_REPORT.md         # cross-run comparison of all runs so far
│   └── aggregate_benchmarks.csv
└── indexes/                        # on-disk index cache (reused across runs)
```

The two reports you will read most often:

- **`reports/<run_id>/REPORT.md`** — the full write-up of a single run, with one
  section per suite, embedded figures, and findings.
- **`reports/_aggregate/AGGREGATE_REPORT.md`** — compares the current run against
  every previous run.

### Rebuilding reports without re-running experiments

The report builder can regenerate `REPORT.md` (for the most recent run) and the
aggregate report from the persisted `_suite_summary.json` files. No suite is
re-executed:

```bash
uv run experiments/reports/report_builder.py
```

## Expected console output

A successful retrieval-mode run logs roughly the following. The `run_id` and
timings differ per machine, and your corpus and query counts depend on the flags
you passed:

```text
======================================================================
Starting run run_20260527_144558_fbf83b
======================================================================
Corpora: ['n10', 'n50', 'n100', 'n1000']
Loaded corpus n10: 42 chunks
Loaded corpus n50: 210 chunks
Loaded corpus n100: 415 chunks
Loaded corpus n1000: 4180 chunks
Generated 60 queries for n10
Generated 60 queries for n50
Generated 60 queries for n100
Generated 60 queries for n1000
Generator disabled or unreachable: generation disabled via configuration (env/CLI)
Running 21 suites: ['s01_retriever_corpus_scaling', 's02_topk_sensitivity', ...]
Starting suite s01_retriever_corpus_scaling (How retrieval quality scales ...)
Suite s01_retriever_corpus_scaling finished in 18.42s
...
Suite s10_context_pollution finished in 0.05s        # skipped: generator unavailable
...
======================================================================
Run run_20260527_144558_fbf83b complete
Report:    experiments/outputs/reports/run_20260527_144558_fbf83b/REPORT.md
Manifest:  experiments/outputs/logs/run_20260527_144558_fbf83b/manifest.json
Dashboard: experiments/outputs/reports/run_20260527_144558_fbf83b/dashboard.png
Aggregate: experiments/outputs/reports/_aggregate/AGGREGATE_REPORT.md
======================================================================
```

With `--enable-generation` and a reachable model you instead see
`Generator available: model=mistral`, and suites `s10`, `s11`, `s17` report a
non-trivial duration instead of an immediate skip.

The process exit code is `0` on success, `1` if every suite failed, and `2` if
no corpus could be loaded.

## Typical runtimes

Runtimes vary widely with hardware, corpus size, and the model cache state, so
treat the following as rough orientation rather than exact figures:

- **First run dominates on downloads.** The first invocation downloads the
  embedding models (and, with generation enabled, the Ollama model ≈ 4 GB).
  After that, the Hugging Face cache and the on-disk index cache under
  `experiments/outputs/indexes/` are reused.
- **Warm retrieval-mode full run.** On a typical modern CPU-only laptop, a warm
  run across all default corpora is on the order of several minutes to roughly
  twenty minutes; the `n1000` corpus and the dense/hybrid suites dominate.
- **Generation suites.** Their cost is essentially Ollama's throughput on your
  hardware multiplied by the number of queries — anywhere from a few minutes to
  much longer.

To keep runs short while iterating, reduce the work:

```bash
# small corpora, few queries, a single suite
uv run experiments/run_all_experiments.py \
    --corpora n10,n50 \
    --n-queries 10 \
    --only-suites s01
```

The benchmark iteration counts are also intentionally small by default
(warmup/measured iterations); raise them via the `EXPERIMENTS_BENCH_*`
environment variables for more stable numbers at the cost of runtime.

## Reproducibility checklist

1. Fix the seed (`--seed`, default `1337`) and the query count (`--n-queries`).
2. Pin the corpora (`--corpora`) so the matrix is identical.
3. Record the `run_id` printed at the end — it names every artefact for that run.
4. Keep `uv.lock` unchanged so the dependency graph is identical.
5. For generation runs, pin the Ollama model tag (default `mistral`).
