# Deployment

The RAG Toolkit is designed for **resource-constrained local execution** on a
single Docker host: a developer laptop, a classroom workstation, or a single
VM. It is not designed or tested for multi-node HPC deployment.

## Supported target: single Docker host

| Requirement       | Value                                                  |
|-------------------|--------------------------------------------------------|
| Container engine  | Docker Engine ≥ 24 with Compose v2                     |
| OS                | Linux or WSL2 with access to `/var/run/docker.sock`    |
| Disk              | ≈ 10 GB free (notebook image + Mistral + HF cache)     |
| RAM               | ≥ 8 GB recommended for Mistral 7B Q4 inference         |
| GPU               | Not required; optional, see below                      |
| Network           | None at runtime; image pull and HF cache need outbound |

## Standard start (CPU-only)

```bash
docker compose up --build
```

Compose dependencies enforce the order:

1. `notebook-builder` builds and tags `rag-notebook:0.2.0`, then exits 0.
2. `ollama` becomes healthy.
3. `ollama-init` pulls `mistral:latest` on first run, then exits 0.
4. `hub` starts.

Open <http://localhost:8000>.

## Optional GPU profile

See [GPU Experiments](gpu-experiments.md) for the full opt-in workflow. The
short version:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

GPU is an opt-in experiment, not a different supported deployment.

## Rebuilding the notebook image

After editing `Dockerfile.notebook`, `Dockerfile.notebook.gpu`, `pyproject.toml`,
`uv.lock`, or the `rag` package:

```bash
docker compose build notebook-builder           # CPU
# or
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build notebook-builder
docker compose up -d hub
```

Currently running notebook containers continue to use the previous image; new
spawns pick up the rebuilt one.

## Persistent state

| Volume                         | Contents                                |
|--------------------------------|-----------------------------------------|
| `jupyterhub-data`              | JupyterHub state (cookie secret, DB)    |
| `ollama-data`                  | Pulled Ollama models                    |
| `jupyterhub-work-{username}`   | Per-user `~/workspace`                  |
| `hf-cache`                     | HuggingFace model cache                 |

## Out of scope

Multi-node HPC deployment (Slurm + Apptainer/Singularity, Kubernetes with
`KubeSpawner` or `BatchSpawner`) is **out of scope** for this toolkit. The
current `DockerSpawner`-based design relies on access to `/var/run/docker.sock`
on a single host and does not transfer to a Slurm scheduler. A migration would
require:

- Replacing `DockerSpawner` with `BatchSpawner` / `SlurmSpawner` /
  `KubeSpawner`.
- Converting the notebook Docker image to an Apptainer SIF (e.g.
  `apptainer build rag-notebook.sif docker://rag-notebook:0.2.0`).
- Moving from Docker named volumes to a shared cluster filesystem.
- Adapting Ollama deployment to whatever the cluster permits.

This is recorded as future work, not as a supported configuration.
