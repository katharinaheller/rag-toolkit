#!/usr/bin/env bash
#
# /opt/ops/run-benchmarks.sh
#
# Runs the full GPU experiment + benchmark pipeline inside the
# rag-benchmark-runner container. Invoked from the host via:
#
#   docker exec -i rag-benchmark-runner bash /opt/ops/run-benchmarks.sh
#
# All EXPERIMENTS_* defaults come from the container environment
# (docker-compose.gpu.yml); the lines below only fill gaps so the script is
# also safe to run by hand inside `docker exec -it ... bash`.

set -euo pipefail

: "${EXPERIMENTS_OUTPUT_ROOT:=/opt/experiment-outputs}"
: "${EXPERIMENTS_CACHE_ROOT:=${EXPERIMENTS_OUTPUT_ROOT}/indexes}"
: "${EXPERIMENTS_DATA_ROOT:=/opt/data/documents}"
: "${EXPERIMENTS_ENABLE_GPU_BENCHMARKS:=1}"

export PYTHONPATH="/opt:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export EXPERIMENTS_OUTPUT_ROOT EXPERIMENTS_CACHE_ROOT EXPERIMENTS_DATA_ROOT
export EXPERIMENTS_ENABLE_GPU_BENCHMARKS
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

mkdir -p "${EXPERIMENTS_OUTPUT_ROOT}" "${MPLCONFIGDIR}"

echo "==================================================================="
echo "[run-benchmarks] runtime check"
python - <<'PY'
import sys
print("python   :", sys.version.split()[0])
try:
    import torch
    print("torch    :", torch.__version__)
    print("cuda     :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device   :", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch    : import failed:", exc)
try:
    import pynvml  # noqa: F401
    print("pynvml   : available")
except Exception as exc:
    print("pynvml   : not available:", exc)
import experiments  # noqa: F401
print("experiments package import: OK")
PY
echo "[run-benchmarks] output root : ${EXPERIMENTS_OUTPUT_ROOT}"
echo "[run-benchmarks] data root   : ${EXPERIMENTS_DATA_ROOT}"
echo "==================================================================="

echo "[run-benchmarks] launching full experiment pipeline..."
python -m experiments.run_all_experiments "$@"

echo "[run-benchmarks] rebuilding per-run + aggregated reports..."
python -m experiments.reports.report_builder || true

echo "==================================================================="
echo "[run-benchmarks] DONE. Artefacts under: ${EXPERIMENTS_OUTPUT_ROOT}"
echo "  per-run report : ${EXPERIMENTS_OUTPUT_ROOT}/reports/<run_id>/REPORT.md"
echo "  aggregate      : ${EXPERIMENTS_OUTPUT_ROOT}/reports/_aggregate/AGGREGATE_REPORT.md"
echo "==================================================================="
