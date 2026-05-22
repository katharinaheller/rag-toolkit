#!/usr/bin/env bash
#
# /opt/ops/gpu-smoke.sh
#
# Fast sanity check that the runner sees the GPU and that the experiments
# package imports. Does not run benchmarks. Invoked via:
#
#   docker exec -i rag-benchmark-runner bash /opt/ops/gpu-smoke.sh

set -euo pipefail
export PYTHONPATH="/opt:${PYTHONPATH:-}"

python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda_available =", ok)
if ok:
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("CUDA NOT AVAILABLE inside benchmark-runner")
from experiments.core.gpu_hardware import collect_hardware_metadata
hw = collect_hardware_metadata()
print("primary gpu (metadata):", hw.primary_gpu_name)
print("SMOKE OK")
PY
