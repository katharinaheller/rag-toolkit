# RAG Toolkit

A local Retrieval-Augmented Generation (RAG) toolkit with clean, testable
layers for ingestion, embedding, indexing, retrieval, generation, and
evaluation. It ships three things in one repository:

1. **`rag/`** — an importable, modular Python library. Each layer has a narrow
   contract and can be swapped, tested, and benchmarked independently.
2. **`experiments/`** — a research framework that runs a matrix of retrieval,
   generation, scaling, and resource benchmarks and produces Markdown reports.
3. **A JupyterHub platform** — a reproducible, per-user notebook environment
   (JupyterHub + DockerSpawner + Ollama) for interactive exploration.

Generation runs locally via [Ollama](https://ollama.com/); embeddings run
locally via sentence-transformers / FlagEmbedding. No external APIs are
required at runtime.

## Two ways to use this repository

| You want to…                                              | Start here                                  |
|-----------------------------------------------------------|---------------------------------------------|
| Run experiments / use the `rag` library on your machine   | [Quick start (uv)](#quick-start-uv) — the primary path |
| Reproduce the full experiment matrix and reports          | [Running experiments](#running-experiments) |
| Use the interactive notebook platform                     | [JupyterHub platform (Docker)](#jupyterhub-platform-docker) |

The library and the experiment framework are driven with
[uv](https://docs.astral.sh/uv/) and need no Docker. The JupyterHub platform is
a separate, optional deployment.

## Run all commands from the repository root

Every command in this README is executed from the repository root — the folder
that contains `pyproject.toml`:

```bash
cd rag-toolkit
```

## Prerequisites

**For the library and experiments (uv path — recommended):**

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) ≥ 0.5
- For generation suites: a running [Ollama](https://ollama.com/) instance with a
  pulled model (optional — retrieval works without it)
- For the gated embedding model `google/embeddinggemma-300m`: a Hugging Face
  access token (see [Embedding models](#embedding-models-and-hugging-face))

**For the JupyterHub platform (Docker path — optional):**

- Docker Engine ≥ 24 with Compose v2
- Linux or WSL2 host with access to `/var/run/docker.sock`
- ≈ 10 GB free disk space (notebook image, Ollama model, HF cache)

## Quick start (uv)

Install the project (the `rag` library plus everything the experiment framework
needs) into a managed virtual environment:

```bash
uv sync
```

`uv sync` installs the runtime dependencies plus the default `dev` group. Two
further groups are available and are installed explicitly when needed:

```bash
uv sync --group dev      # test/lint tooling (pytest, pylint, hypothesis, …)
uv sync --group docs     # mkdocstrings — required to build the documentation
```

Run the tests to confirm the environment:

```bash
uv run pytest
```

A minimal end-to-end use of the library:

```python
from rag.evaluation import (
    EvaluationConfig, EvaluationRunner, EvaluationReport, dataset_from_dicts,
)

config = EvaluationConfig(mode="end_to_end", top_k=5)
examples = dataset_from_dicts([
    {
        "query": "What is RAG?",
        "expected_answer": "Retrieval-Augmented Generation",
        "relevant_document_ids": ["doc_abc123"],
    }
])
runner = EvaluationRunner(config, retriever=my_retriever, strategy=my_strategy)
result = runner.run(examples)
print(EvaluationReport(result).summary())
```

## Running experiments

The `experiments/` framework runs a matrix of suites against the `rag` library
and writes a Markdown report, a dashboard image, and a run manifest. The
[Running Experiments](docs/experiments.md) guide documents every suite, option,
and output in full; the essentials are below.

There are two operating modes. The only difference is whether the **generation**
(LLM) suites run.

### Retrieval mode (default — no Ollama)

Computes everything that does not need a language model: retrieval quality,
scaling, latency, embedding comparisons, robustness, stability, and the resource
benchmarks. The three generation suites (`s10`, `s11`, `s17`) detect that no
generator is available and record a clean `skipped` status.

```bash
uv run experiments/run_all_experiments.py
```

### Generation mode (needs Ollama + a model)

Generation suites need a local model served by Ollama. Two things must be true:
Ollama must be reachable, **and** the model must be installed. **Ollama being
reachable does not mean the model is present** — pull it once:

```bash
ollama pull mistral
```

Then enable generation explicitly:

```bash
uv run experiments/run_all_experiments.py --enable-generation
```

### Selected suites

Restrict the run to specific suites with `--only-suites` (comma-separated short
ids like `s05`, or full keys). This is the fast way to iterate or to reproduce a
single result:

```bash
uv run experiments/run_all_experiments.py \
    --enable-generation \
    --only-suites s10,s11,s17
```

### Where results go

Artefacts are written under `experiments/outputs/`, namespaced by a unique
`run_id`. The two reports you will read most often:

- `experiments/outputs/reports/<run_id>/REPORT.md` — the full per-run write-up.
- `experiments/outputs/reports/_aggregate/AGGREGATE_REPORT.md` — a comparison
  across all runs so far.

To rebuild the reports from stored results without re-running any suite:

```bash
uv run experiments/reports/report_builder.py
```

See [Running Experiments](docs/experiments.md) for the complete output layout,
expected console output, runtime guidance, and the reproducibility checklist.

### Embedding models and Hugging Face

Even a retrieval-only run downloads embedding models, because the dense and
hybrid retrievers depend on them:

- `BAAI/bge-m3` — openly available, no token required.
- `google/embeddinggemma-300m` — **gated**; accept its license on Hugging Face
  and authenticate before first use.

Export a read token (create one at <https://huggingface.co/settings/tokens>)
before running suites that use the `gemma` embedder:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Only the pure sparse suites (BM25, TF-IDF) avoid model downloads entirely.

## Command line interface

The repository contains four `argparse` CLIs plus a standalone report builder.
The [Command Line Interface](docs/cli.md) page documents every option of every
tool; the summary:

| Command                                     | Purpose                                                       |
|---------------------------------------------|---------------------------------------------------------------|
| `experiments/run_all_experiments.py`        | Run the full experiment matrix (main entry point)             |
| `experiments/reports/report_builder.py`     | Rebuild Markdown reports from stored results (no re-run)       |
| `experiments/tools/rescore_generation.py`   | Re-score stored generation records offline (no Ollama)        |
| `analysis/analyse_results.py`               | Print comparison tables from stored evaluation results        |
| `analysis/plot_latency.py`                  | Generate publication-quality figures from stored results      |

Every tool responds to `--help`, for example:

```bash
uv run experiments/run_all_experiments.py --help
```

> **Note:** `run_all_experiments.py` writes to `experiments/outputs/`, while the
> `analysis/` tools read from `results/eval/` and `results/benchmarks/` (produced
> by the `rag.evaluation` JSONL stores, e.g. from the notebooks). These are
> different locations — see the [CLI reference](docs/cli.md) for details.

## The library (`rag` package)

```
rag/
  ingestion/      load, clean, chunk documents
  embedding/      compute dense and sparse vectors
  indexing/       build dense (BruteForce, FAISS) and sparse indexes
  retrieval/      score and rank candidates (dense, BM25, TF-IDF, learned, hybrid)
  generation/     prompt the local LLM via Ollama
  evaluation/     metrics, monitors, benchmarks, reports
  logging/        structured logging helpers
```

Each layer follows the same conventions: immutable frozen dataclasses for
configuration, deterministic IDs (SHA-256 over natural keys or content), and
streaming pipelines (generators, no full-corpus loads). See the
[API reference](docs/api.md) for the public surface of every layer.

## JupyterHub platform (Docker)

The optional platform combines JupyterHub, DockerSpawner, and Ollama to provide
reproducible, per-user notebook environments with CPU-only PyTorch and a local
Mistral model. The [Deployment guide](docs/deployment.md) covers this in full;
the essentials follow.

### Architecture

Four Compose services orchestrate the platform:

- **`notebook-builder`** — builds and tags `rag-notebook:0.2.0`. Its entrypoint
  is overridden with `true` so the container exits immediately after the image
  is materialized. `hub` depends on its successful completion.
- **`ollama`** — runs `ollama/ollama:0.6.8` with a persistent model volume and a
  healthcheck on `ollama ps`.
- **`ollama-init`** — waits for the API, pulls `mistral:latest` once, then exits.
- **`hub`** — runs JupyterHub 4.0 with `DockerSpawner` and `NativeAuthenticator`.
  Spawns one notebook container per user on the `rag-toolkit_backend` network.

Notebook containers receive the following mounts:

| Source                          | Destination                       | Mode |
|---------------------------------|-----------------------------------|------|
| `jupyterhub-work-{username}`    | `/home/jovyan/workspace`          | rw   |
| `./notebooks` (host)            | `/home/jovyan/shared/notebooks`   | ro   |
| `./data` (host)                 | `/home/jovyan/shared/data`        | ro   |
| `hf-cache`                      | `/srv/hf-cache`                   | rw   |

The hub resolves the host-side paths for the shared mounts by inspecting its own
container via `/var/run/docker.sock` at startup (see `jupyterhub_config.py`).
DockerSpawner forwards the same host paths to each user container.

### Starting the platform

```bash
docker compose up -d --build
```

Order of operations enforced by Compose dependencies:

1. `notebook-builder` builds and tags `rag-notebook:0.2.0`, then exits 0.
2. `ollama` becomes healthy.
3. `ollama-init` pulls `mistral:latest` on first run, then exits 0.
4. `hub` starts once both prerequisite steps have completed successfully.

Open <http://localhost:8000>. Signup is open by default; the account whose
username matches `JUPYTERHUB_ADMIN` (default `admin`) is granted admin rights. On
first login, DockerSpawner launches a `rag-notebook:0.2.0` container and
redirects to `/lab/tree/shared/notebooks`.

### Notebook workflow

Inside a spawned container:

- **`/home/jovyan/workspace`** — writable, per-user, persistent across restarts.
  Place personal notebooks under `workspace/notebooks` and intermediate
  artefacts under `workspace/data/{raw,processed}`. These subdirectories are
  seeded from the image on first spawn and topped up by `init-workspace.sh` on
  every spawn (new subdirectories are created; existing files are never
  overwritten).
- **`/home/jovyan/shared/notebooks`** and **`/home/jovyan/shared/data`** —
  read-only reflections of the host-side `./notebooks` and `./data` directories.
  Use them for reference material and copy into `workspace/` before editing.

Never write to `shared/`; it is mounted read-only. The path
`/home/jovyan/workspace/shared` deliberately does not exist.

The shipped notebooks walk through the layers in order:

```
notebooks/
  01_rag_ingestion_layer.ipynb
  02_rag_embedding_layer.ipynb
  03_rag_indexing_layer.ipynb
  04_rag_retrieval_layer.ipynb
  05_rag_generation_layer.ipynb
  06_rag_gpu_pipeline.ipynb        # optional GPU benchmark — safe on CPU-only
  07_rag_evaluation.ipynb
```

### Rebuilding the notebook image

After editing `Dockerfile.notebook`, `pyproject.toml`, `uv.lock`, or the `rag`
package:

```bash
docker compose build notebook-builder
docker compose up -d hub
```

Currently running notebook containers continue to use the previous image; new
spawns pick up the rebuilt one. `pull_policy = "never"` is set intentionally —
the image must exist locally.

### Ollama

The `ollama` service exposes the API at `http://ollama:11434` on the
`rag-toolkit_backend` network — reachable from notebook containers under the
hostname `ollama`. Models live in the `ollama-data` named volume and survive
container recreation.

`ollama-init` pulls the models listed in its `OLLAMA_MODELS` environment variable
(default: `mistral:latest`). To add more models, edit `OLLAMA_MODELS` in
`docker-compose.yml` (space-separated) and run `docker compose up -d ollama-init`,
or exec into the running container: `docker exec -it ollama ollama pull <model>`.

### GPU profile (opt-in)

The platform is CPU-only by default. An optional GPU profile is available for
performance experiments; it is not required and changes no results, only their
speed. See [GPU Experiments](docs/gpu-experiments.md) for the full opt-in
workflow.

## Documentation

The documentation is built with MkDocs (Material theme) and `mkdocstrings`,
which renders the API reference from the source docstrings.

`mkdocs` and `mkdocs-material` are runtime dependencies, but `mkdocstrings[python]`
lives in the `docs` dependency group. A fresh clone therefore needs that group
installed before MkDocs can resolve the plugin referenced in `mkdocs.yml`:

```bash
uv sync --group docs
uv run --group docs mkdocs serve     # http://127.0.0.1:8000
uv run --group docs mkdocs build     # static site in ./site
```

Plain `uv run mkdocs serve` on a fresh clone fails with a plugin-not-found error
because `mkdocstrings` is not part of the default sync. Always pass
`--group docs` for documentation work.

> The hub also binds port 8000. When previewing the docs while the platform is
> running, use `mkdocs serve -a 127.0.0.1:8001` (or stop the hub first).

## Development

Host-side environment for editing the `rag` package, running tests, or building
docs:

```bash
uv sync                  # runtime + default (dev) group
uv sync --group dev      # explicit dev tooling (already part of uv sync)
uv sync --group docs     # add mkdocstrings for the docs site
```

The `dev` group provides `pytest`, `pylint`, `hypothesis`, `pyfakefs`, and
Sphinx-based alternatives kept for ad-hoc reports (not used by the active MkDocs
setup). The `docs` group is the minimal set required to build the documentation
site.

Run tests:

```bash
uv run pytest
```

The notebook image is built from `pyproject.toml` and `uv.lock` via `uv export`,
so locking changes on the host directly influence the next image build. Commit
`uv.lock` alongside `pyproject.toml`.

## Repository layout

```
.
├── pyproject.toml             # Python package + dependency groups
├── uv.lock                    # locked dependency graph
├── mkdocs.yml                 # documentation site config
├── rag/                       # importable Python library (the six layers)
├── experiments/               # research framework + suites + entry point
├── analysis/                  # standalone table/figure CLIs
├── notebooks/                 # layer-by-layer notebooks (shared, read-only in containers)
├── data/                      # corpora + evaluation datasets (read-only in containers)
├── docs/                      # MkDocs sources
├── ops/                       # operator shell scripts for the GPU stack
├── tests/                     # unit, integration, and e2e tests
├── uml/                       # generated architecture diagrams
├── docker-compose.yml         # JupyterHub platform orchestration
├── docker-compose.gpu.yml     # opt-in GPU override
├── Dockerfile                 # JupyterHub image (rag-hub:4.0)
├── Dockerfile.notebook        # per-user notebook image (rag-notebook:0.2.0)
├── Makefile / benchmark.ps1   # operator helpers (GPU benchmark stack)
└── jupyterhub_config.py       # DockerSpawner + auth configuration
```

## Troubleshooting

**Hub exits with "Cannot inspect hub container" / "Required bind mount not found".**
The hub inspects its own container via the Docker socket to discover the host
paths behind its bind mounts. Make sure `/var/run/docker.sock` is mounted and the
hub is started through `docker compose up` (not `docker run`).

**DockerSpawner cannot find `rag-notebook:0.2.0`.**
`pull_policy = "never"` is intentional — the image must exist locally. Run
`docker compose build notebook-builder` and verify with
`docker images | grep rag-notebook`.

**Spawn fails with a network error.**
`DOCKER_NETWORK_NAME` must match the actual Compose network. It is pinned to
`rag-toolkit_backend`; do not rename the project or network without updating both
`docker-compose.yml` and `jupyterhub_config.py`.

**`mkdocs serve` fails with "plugin not found: mkdocstrings".**
Run `uv sync --group docs` and invoke MkDocs with
`uv run --group docs mkdocs serve`.

**Experiment generation suites are always skipped.**
Generation suites need Ollama reachable **and** the model pulled. Run
`ollama pull mistral` and pass `--enable-generation`. A reachable Ollama with no
model still results in a skip.

**`analyse_results.py` reports no results after running the experiments.**
The experiment framework writes to `experiments/outputs/`, but the analysis tools
read from `results/eval/` and `results/benchmarks/`. Point them at the right
directory with `--eval-dir` / `--benchmark-dir`.

**Workspace appears empty for a returning user.**
Each user has an isolated `jupyterhub-work-{username}` named volume. New users see
the image-seeded skeleton; existing users see whatever they last saved.
`init-workspace.sh` only creates missing subdirectories — it never overwrites
files.

**Ollama model missing (platform).**
Inspect `docker compose logs ollama-init`. The init container exits after
pulling; if it failed (e.g. transient network error), rerun
`docker compose up -d ollama-init`.
