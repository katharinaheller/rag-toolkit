# GPU Experiments (opt-in)

The RAG Toolkit is **CPU-only by default**. The default Compose stack does not
allocate GPUs, the default notebook image strips GPU packages from the
dependency lock, and `torch` is installed from the PyTorch CPU index. CPU-only
is the standard, documented, supported configuration.

This page describes the **optional** GPU path used for performance comparison
experiments. It is not required and not part of the standard student
workflow.

## Why CPU is the default

- **Reproducibility.** Identical behaviour on every machine that runs Docker;
  no driver version drift.
- **Low hardware requirements.** Works on consumer laptops without a discrete
  NVIDIA GPU.
- **No driver dependency.** No host-side CUDA, no `nvidia-container-toolkit`,
  no kernel module compatibility concerns.
- **Suitable for the teaching context.** Most students do not own
  CUDA-capable hardware.
- **More stable local execution.** GPU passthrough into Docker adds failure
  modes (mismatched driver, missing runtime, container toolkit
  misconfiguration) that obscure issues in the RAG pipeline itself.

## Why GPU is opt-in only

- Not every machine has CUDA / NVIDIA hardware.
- GPU drivers and `nvidia-container-toolkit` make Docker setups more fragile.
- GPU is only useful here for **performance evaluation**, not correctness —
  every result the toolkit produces on GPU is reproducible on CPU.

## Prerequisites

Required on the host machine, *only* if you choose to enable the GPU profile:

1. An NVIDIA GPU.
2. A recent NVIDIA driver compatible with CUDA 12.1.
3. [`nvidia-container-toolkit`][toolkit] installed and the Docker daemon
   restarted afterwards.

[toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

A quick host-side check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If either command fails, the GPU profile will not work — use the default
CPU-only stack.

## Starting the GPU stack

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The override file does three things and nothing else:

1. Rebuilds `notebook-builder` from `Dockerfile.notebook.gpu` and tags it
   `rag-notebook-gpu:0.2.0`.
2. Adds an NVIDIA device reservation to the `ollama` service.
3. Tells the hub to spawn `rag-notebook-gpu:0.2.0` instead of the CPU image.

The base `docker-compose.yml` is unchanged. A user who never invokes the
override will never get any GPU code paths.

## Starting the CPU stack (default)

```bash
docker compose up --build
```

This remains the supported standard workflow. Nothing in this document
changes that.

## Running the GPU benchmark notebook

Inside JupyterLab, open
`shared/notebooks/06_gpu_optional_benchmark.ipynb`. The notebook is designed
to be safe to run regardless of the active stack: on the CPU stack it
reports "no GPU available" and skips the GPU half cleanly.

The notebook checks, in order:

1. Is `pynvml` importable?
2. Is the NVIDIA driver visible inside the container (`nvidia-smi`)?
3. Does the toolkit's `GpuMonitor` report a usable device?
4. Does `torch.cuda.is_available()` return True?
5. A small synthetic embedding benchmark on CPU and (if available) GPU.

A missing GPU is **not an error** — it is reported as a clear "skipped"
state and the notebook continues.

## Programmatic GPU monitoring

`GpuMonitor` is safe to call on any machine:

```python
from rag.evaluation.monitors.gpu_monitor import GpuMonitor

monitor = GpuMonitor()
info = monitor.info()

if info.available:
    print(f"GPU: {info.name}, {info.memory_used_mb:.0f}/{info.memory_total_mb:.0f} MB")
else:
    print(f"GPU not available: {info.error}")
```

To include GPU metrics in a benchmark's resource snapshots:

```python
from rag.evaluation import EvaluationConfig

config = EvaluationConfig(
    capture_resources=True,
    gpu_monitoring=True,  # safe on CPU-only — yields None metrics, no error
)
```

When no GPU is reachable, the snapshot's `gpu_*` fields are `None`. This is
not a failure mode; it is the documented behaviour.

## How GPU benchmarking is interpreted

- CPU and RAM metrics are always collected, with or without a GPU.
- GPU metrics are collected only when a GPU is reachable.
- A missing GPU is not a benchmark failure.
- Cross-hardware comparisons are not portable; only relative numbers from the
  same host are meaningful.

## What this is *not*

This is **not** an HPC deployment path. The toolkit's `DockerSpawner` setup is
single-host. Running on an actual HPC cluster (Slurm + Apptainer,
Kubernetes + `KubeSpawner`, etc.) is recorded as future work in the
[Performance Tuning Guide](../rag/evaluation/documentation/performance_tuning.md#beyond-a-single-host-outlook),
not as a supported configuration.
