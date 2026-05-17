# RAG Experiments Framework

A thesis-grade evaluation suite for the `rag-toolkit` project. Runs fifteen
research-style experiments end-to-end with a single command, produces raw and
aggregated data, publication-quality figures, and an auto-generated Markdown
report.

This package is read-only with respect to the rest of the toolkit. It wraps
the real `rag.*` modules (no fake APIs, no stubs) and stores all artefacts
under `experiments/outputs/<run_id>/`.

---

## 1. What you get

For every run the framework produces:

```
experiments/outputs/
  raw/<run_id>/<suite>/*.jsonl          per-query retrieval & generation records
  aggregated/<run_id>/<suite>/*.csv     headline tables (one row per cell of the matrix)
  figures/<run_id>/<suite>/*.png        scaling curves, Pareto fronts, heatmaps, …
  reports/<run_id>/REPORT.md            auto-generated 15-section Markdown report
  reports/<run_id>/dashboard.png        four-panel summary figure
  logs/<run_id>/run.log                 full stdout / stderr capture
  logs/<run_id>/manifest.json           machine-readable run summary
  indexes/<corpus>/<retriever>/         cached indexes (reused across runs)
```

The fifteen suites:

| Key | Question answered |
|-----|--------------------|
| `s01_retriever_corpus_scaling` | Which retriever scales best from n10 to n1000? Where are the crossovers? |
| `s02_topk_sensitivity` | At which k does recall saturate? At which k does context pollution begin? |
| `s03_query_type_comparison` | Which retriever wins on each of the five query types? |
| `s04_embedding_comparison` | EmbeddingGemma vs BGE-M3, head-to-head. |
| `s05_latency_quality_pareto` | Which retrievers are Pareto-optimal on latency × recall? |
| `s06_retrieval_overlap` | Which retrievers retrieve complementary evidence? |
| `s07_hybrid_fusion_sweep` | What is the best dense/sparse fusion weight? |
| `s08_noise_robustness` | Which retriever degrades least under noisy queries? |
| `s09_stability` | How stable are results across repeated runs (CIs, rank stability)? |
| `s10_context_pollution` | How does faithfulness drop as gold chunks are replaced with distractors? |
| `s11_long_context` | How does generation quality change as top_k grows the context? |
| `s12_failure_taxonomy` | What is the dominant failure mode per retriever (no_hit / low_rank / polluted)? |
| `s13_chunk_relevance_decay` | How quickly does the per-rank hit rate decay? |
| `s14_throughput_resources` | Latency p50/p95, throughput and RSS per retriever × corpus. |
| `s15_query_conditioned` | Best retriever per query type and the oracle-routing upper bound. |

---

## 2. Installation

Inside the `rag-toolkit` project root (Python 3.12):

```bash
pip install -r requirements.txt        # if you maintain one
# Optional extras the framework will use when present:
pip install psutil                     # memory snapshots in suite 14
pip install faiss-cpu                  # dense ANN backend on n1000
pip install matplotlib                 # figures (the framework still runs without)
```

The framework itself has no third-party dependencies beyond what `rag.*`
already needs. Plotting, memory measurement and FAISS are detected at
runtime; suites degrade gracefully when any of them is missing.

Copy the entire `experiments/` directory into `C:\rag-toolkit\` so the layout
becomes:

```
C:\rag-toolkit\
  rag\
  data\documents\{n10,n50,n100,n1000}\
  experiments\
    configs\ adapters\ core\ metrics\ storage\ suites\ visualisation\ reports\
    run_all_experiments.py
    README.md
```

---

## 3. One-command usage

From the project root:

```bash
python -m experiments.run_all_experiments
```

That is the canonical invocation. Equivalent direct call:

```bash
python experiments/run_all_experiments.py
```

A typical first run on n10/n50/n100 finishes in a few minutes; n1000 with
dense retrieval and embeddings is the dominant cost (~10–40 min on CPU,
amortised over subsequent runs via the index cache).

### Useful flags

```bash
# Only the cheap, fast suites:
python -m experiments.run_all_experiments --only-suites s01,s02,s05,s06

# Restrict to two corpora and 30 queries each:
python -m experiments.run_all_experiments --corpora n10,n100 --n-queries 30

# Force generation off / on (default = read EXPERIMENTS_ENABLE_GENERATION):
python -m experiments.run_all_experiments --disable-generation
python -m experiments.run_all_experiments --enable-generation
```

---

## 4. Configuration via environment variables

Everything is overridable without editing code:

| Variable | Default | Effect |
|----------|---------|--------|
| `EXPERIMENTS_DATA_ROOT` | `<repo>/data/documents` | Where `n10`, `n50`, `n100`, `n1000` live. |
| `EXPERIMENTS_OUTPUT_ROOT` | `<repo>/experiments/outputs` | All artefacts written here. |
| `EXPERIMENTS_CACHE_ROOT` | `<output>/indexes` | Reused across runs. |
| `EXPERIMENTS_ENABLE_GENERATION` | `false` | Enable generation-dependent suites. |
| `EXPERIMENTS_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL. |
| `EXPERIMENTS_OLLAMA_MODEL` | `mistral` | Ollama model tag. |
| `EXPERIMENTS_OLLAMA_TIMEOUT` | `120.0` | HTTP timeout (s) for one generation call. |
| `EXPERIMENTS_N_QUERIES` | `60` | Synthetic queries per corpus (stratified across 5 types). |
| `EXPERIMENTS_SEED` | `1337` | Deterministic seed for query generation. |
| `EXPERIMENTS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, … |

On Windows:

```powershell
$env:EXPERIMENTS_ENABLE_GENERATION = "1"
$env:EXPERIMENTS_OLLAMA_URL = "http://localhost:11434"
python -m experiments.run_all_experiments
```

---

## 5. Reproducibility

Three independent deterministic layers:

1. **Query generation** uses a single seeded `random.Random(seed)` instance per
   corpus. Same seed + same chunks → identical query set.
2. **Chunk caching** writes each corpus's chunks to JSONL (under
   `outputs/indexes/chunks/`) keyed by the corpus directory. Re-runs skip the
   ingestion pipeline.
3. **Embedding & index caching** stores embeddings (JSONL) keyed by a
   chunk-id digest and writes a `.built` marker per index. Switching seeds
   does **not** invalidate the index (chunks are unchanged), only the queries.

The stability suite (s09) re-runs the same retrieval 5 times to expose any
remaining non-determinism (mostly tie-breaking on sparse and FAISS HNSW).

---

## 6. Generation suites

Suites 10 and 11 require a running local LLM (default Ollama / Mistral). The
generation adapter probes `GET /api/tags` once at start-up. If unreachable:

* `s10_context_pollution` is marked `skipped` with a clear note in the report.
* `s11_long_context` is marked `skipped` likewise.
* All other suites are unaffected.

To enable:

```bash
ollama serve
ollama pull mistral
EXPERIMENTS_ENABLE_GENERATION=1 python -m experiments.run_all_experiments
```

---

## 7. Troubleshooting

**FAISS not installed.** The indexing adapter falls back to `brute_force` for
all corpora. n1000 dense queries will be noticeably slower (still correct).
`pip install faiss-cpu` to fix.

**matplotlib not installed.** Every plot function silently returns `None`.
Tables/CSV/JSONL outputs are unaffected; the report still renders, just
without figures. `pip install matplotlib` to fix.

**Ollama not reachable / `EXPERIMENTS_ENABLE_GENERATION` unset.** Suites that
need generation mark themselves as `skipped` with a recovery hint. Retrieval
suites are unaffected.

**OOM on n1000.** Lower `EXPERIMENTS_N_QUERIES`, run subsets via
`--only-suites s01,s02,s05,s14`, or shrink the embedding batch size by
editing `configs/default_matrix.py:EmbedderSpec.batch_size`.

**A suite fails with an exception.** The runner catches per-suite exceptions
and records them in the suite's summary with `status="failed"`; other suites
keep running. The traceback is in `outputs/logs/<run_id>/run.log`.

**Slow first run.** First-time embedding and index build is by far the
dominant cost. Subsequent runs reuse the on-disk cache and finish in
minutes.

---

## 8. Extending the framework

Every suite follows the same template:

```python
from experiments.suites.base import Suite

class MyNewSuite(Suite):
    key = "s99_my_question"
    description = "Why does X happen when Y?"

    def run(self):
        ...
        return {
            "figures": [str(fig_path)],
            "tables":  [str(csv_path)],
            "findings": ["Auto-generated string about what the data shows."],
            "rows":    rows,
        }
```

Register it in `experiments/suites/__init__.py:ALL_SUITES` and it is picked
up automatically by `--only-suites` and by the report builder.

Add a new retriever or embedder by appending an `EmbedderSpec` /
`RetrieverSpec` entry in `experiments/configs/default_matrix.py`; the
adapters and every suite consume the list directly.

---

## 9. Citation / context

This framework is the evaluation half of the `rag-toolkit` thesis project.
The retrieval, indexing, generation and metrics infrastructure all live in
`rag.*`. The experiments package is the research-grade harness sitting on
top: it answers research questions, produces figures, and writes the report.

If you find the suites useful, please cite the parent project.
