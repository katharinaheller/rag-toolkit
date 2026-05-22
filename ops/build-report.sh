#!/usr/bin/env bash
#
# /opt/ops/build-report.sh
#
# Rebuilds the per-run REPORT.md (latest run) and the cross-run
# AGGREGATE_REPORT.md from already-persisted artefacts, without re-running any
# experiments. Invoked from the host via:
#
#   docker exec -i rag-benchmark-runner bash /opt/ops/build-report.sh

set -euo pipefail

: "${EXPERIMENTS_OUTPUT_ROOT:=/opt/experiment-outputs}"
export PYTHONPATH="/opt:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export EXPERIMENTS_OUTPUT_ROOT
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

echo "[build-report] output root: ${EXPERIMENTS_OUTPUT_ROOT}"
exec python -m experiments.reports.report_builder "$@"
