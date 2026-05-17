# RAG Toolkit

Local RAG development platform combining JupyterHub, DockerSpawner, and Ollama. Provides reproducible, per-user notebook environments with CPU-only PyTorch and a local Mistral model served by Ollama.

## Architecture

Four Compose services orchestrate the platform:

- **`notebook-builder`** — builds and tags `rag-notebook:0.2.0`. Its entrypoint is overridden with `true` so the container exits immediately after the image is materialized. `hub` depends on its successful completion.
- **`ollama`** — runs `ollama/ollama:0.6.8` with a persistent model volume and healthcheck on `ollama ps`.
- **`ollama-init`** — waits for the API, pulls `mistral:latest` once, then exits.
- **`hub`** — runs JupyterHub 4.0 with `DockerSpawner` and `NativeAuthenticator`. Spawns one notebook container per user on the `rag-toolkit_backend` network.

Notebook containers receive the following mounts:

| Source                          | Destination                       | Mode |
|---------------------------------|-----------------------------------|------|
| `jupyterhub-work-{username}`    | `/home/jovyan/workspace`          | rw   |
| `./notebooks` (host)            | `/home/jovyan/shared/notebooks`   | ro   |
| `./data` (host)                 | `/home/jovyan/shared/data`        | ro   |
| `hf-cache`                      | `/srv/hf-cache`                   | rw   |

The hub resolves the host-side paths for the shared mounts by inspecting its own container via `/var/run/docker.sock` at startup (see `jupyterhub_config.py`). The hub is therefore self-aware about where `./notebooks` and `./data` live on the host — DockerSpawner forwards the same host paths to each user container.

## Repository Layout

```
.
├── docker-compose.yml         # orchestration
├── Dockerfile                 # JupyterHub image (rag-hub:4.0)
├── Dockerfile.notebook        # per-user notebook image (rag-notebook:0.2.0)
├── init-workspace.sh          # startup hook inside notebook containers
├── jupyterhub_config.py       # DockerSpawner + auth configuration
├── pyproject.toml             # Python package + dependency groups
├── uv.lock                    # locked dependency graph
├── mkdocs.yml                 # documentation site config
├── rag/                       # importable Python package
├── notebooks/                 # shared, read-only inside containers
├── data/                      # shared, read-only inside containers
└── docs/                      # mkdocs sources
```

## Prerequisites

- Docker Engine ≥ 24 with Compose v2
- Linux or WSL2 host with access to `/var/run/docker.sock`
- ≈ 10 GB free disk space (notebook image, Ollama model, HF cache)
- `uv` ≥ 0.5 on the host (only required for local documentation builds and host-side development)

## Starting the Platform

```bash
docker compose up -d --build
```

Order of operations enforced by Compose dependencies:

1. `notebook-builder` builds and tags `rag-notebook:0.2.0`, then exits 0.
2. `ollama` becomes healthy.
3. `ollama-init` pulls `mistral:latest` on first run, then exits 0.
4. `hub` starts once both prerequisite steps have completed successfully.

Open <http://localhost:8000>. Signup is open by default; the account whose username matches `JUPYTERHUB_ADMIN` (default `admin`) is granted admin rights. On first login, DockerSpawner launches a `rag-notebook:0.2.0` container and redirects to `/lab/tree/shared/notebooks`.

### Rebuilding the notebook image

After editing `Dockerfile.notebook`, `pyproject.toml`, `uv.lock`, or the `rag` package:

```bash
docker compose build notebook-builder
docker compose up -d hub
```

Currently running notebook containers continue to use the previous image; new spawns pick up the rebuilt one. `pull_policy = "never"` is set intentionally — the image must exist locally.

## Notebook Workflow

Inside a spawned container:

- **`/home/jovyan/workspace`** — writable, per-user, persistent across restarts. Place personal notebooks under `workspace/notebooks` and intermediate artefacts under `workspace/data/{raw,processed}`. These subdirectories are seeded from the image on first spawn and topped up by `init-workspace.sh` on every spawn (new subdirectories added in newer image versions are created; existing files are never overwritten).
- **`/home/jovyan/shared/notebooks`** and **`/home/jovyan/shared/data`** — read-only reflections of the host-side `./notebooks` and `./data` directories. Use them for reference material and copy into `workspace/` before editing.

Never attempt to write to `shared/`; it is mounted read-only. The path `/home/jovyan/workspace/shared` deliberately does not exist — it would be hidden by the workspace volume and would mislead users about which directory is writable.

The notebook image installs `torch==2.7.1+cpu` from the PyTorch CPU index and the `rag` package itself. GPU packages are stripped from the resolved requirements before installation and the build asserts no NVIDIA/CUDA libraries are present.

## Ollama

The `ollama` service exposes the API at `http://ollama:11434` on the `rag-toolkit_backend` network — reachable from notebook containers under the hostname `ollama`. Models live in the `ollama-data` named volume and survive container recreation.

`ollama-init` pulls the models listed in its `OLLAMA_MODELS` environment variable (default: `mistral:latest`). To add more models:

- edit `OLLAMA_MODELS` in `docker-compose.yml` (space-separated) and run `docker compose up -d ollama-init`, **or**
- exec into the running container: `docker exec -it ollama ollama pull <model>`.

## Documentation

Documentation is built with MkDocs (Material theme) and `mkdocstrings` for API reference generation from docstrings.

`mkdocs` and `mkdocs-material` are listed as runtime dependencies in `pyproject.toml`, but `mkdocstrings[python]` lives in the `docs` dependency group. A fresh clone therefore needs that group installed before MkDocs can resolve the plugin referenced in `mkdocs.yml`:

```bash
uv sync --group docs
uv run --group docs mkdocs serve     # http://127.0.0.1:8000
uv run --group docs mkdocs build     # static site in ./site
```

Plain `uv run mkdocs serve` on a fresh clone will fail with a plugin-not-found error because `mkdocstrings` is not part of the default sync. Always pass `--group docs` for documentation work.

> The hub also binds port 8000. When previewing docs while the platform is running, pass `mkdocs serve -a 127.0.0.1:8001` (or stop the hub first).

### API reference

`mkdocstrings` reads the installed `rag` package and renders docstrings into the API section of the documentation site. The `rag` package must therefore be importable in the same environment, which `uv sync --group docs` ensures (it installs project dependencies plus the docs group).

## Development

Host-side environment for editing the `rag` package, running tests, or building docs:

```bash
uv sync                       # runtime + project deps
uv sync --group dev           # add test/lint tooling
uv sync --group docs          # add mkdocstrings for the docs site
```

The `dev` group provides `pytest`, `pylint`, `hypothesis`, `pyfakefs`, and Sphinx-based alternatives kept for ad-hoc reports; they are not used by the active MkDocs setup. The `docs` group is the minimal set required to build the documentation site.

Run tests:

```bash
uv run --group dev pytest
```

The notebook image is built from `pyproject.toml` and `uv.lock` via `uv export`, so locking changes on the host directly influence the next image build. Commit `uv.lock` alongside `pyproject.toml`.

## Troubleshooting

**Hub exits with "Cannot inspect hub container" / "Required bind mount not found".**
The hub inspects its own container via the Docker socket to discover the host paths behind `/hub-mounts/notebooks` and `/hub-mounts/data`. Make sure `/var/run/docker.sock` is mounted and the hub is started through `docker compose up` (not `docker run`) so the bind mounts defined in `docker-compose.yml` are present.

**DockerSpawner cannot find `rag-notebook:0.2.0`.**
`pull_policy = "never"` is intentional — the image must exist locally. Run `docker compose build notebook-builder` and verify with `docker images | grep rag-notebook`.

**Spawn fails with a network error.**
`DOCKER_NETWORK_NAME` must match the actual Compose network. It is pinned to `rag-toolkit_backend` in `docker-compose.yml`; do not rename the project or network without updating both `docker-compose.yml` and `jupyterhub_config.py`.

**`mkdocs serve` fails with "plugin not found: mkdocstrings".**
Run `uv sync --group docs` and invoke MkDocs with `uv run --group docs mkdocs serve`.

**Workspace appears empty for a returning user.**
Each user has an isolated `jupyterhub-work-{username}` named volume. New users see the image-seeded skeleton; existing users see whatever they last saved. `init-workspace.sh` only creates missing subdirectories — it never overwrites files.

**Ollama model missing.**
Inspect `docker compose logs ollama-init`. The init container exits after pulling; if it failed (e.g. transient network error), rerun `docker compose up -d ollama-init`.
