#!/usr/bin/env bash
# /usr/local/bin/start-notebook.d/10-init-workspace.sh
#
# Executed inside the notebook container at startup, before JupyterLab
# is launched, by the Jupyter Docker Stacks start-notebook.sh hook
# mechanism.
#
# PURPOSE
#   Ensure the per-user workspace directory structure exists with correct
#   permissions after the named volume has been mounted at
#   /home/jovyan/workspace.
#
#   On first spawn Docker seeds the volume from the image layer, so
#   workspace/notebooks and workspace/data/* already exist.  On subsequent
#   spawns this script is a safety net: it creates any subdirectories
#   added in a newer image version that the existing volume is missing.
#
# MOUNTS AT RUNTIME
#   /home/jovyan/workspace          named volume  rw  (per user)
#   /home/jovyan/shared/notebooks   host bind     ro  (shared)
#   /home/jovyan/shared/data        host bind     ro  (shared)
#
# INVARIANT
#   Never create /home/jovyan/workspace/shared.  That path sits under the
#   workspace volume mount point, would be hidden once the volume is
#   attached, and misleads users into thinking shared content is writable.

set -euo pipefail

WORKSPACE=/home/jovyan/workspace

mkdir -p \
    "${WORKSPACE}/notebooks" \
    "${WORKSPACE}/data/raw" \
    "${WORKSPACE}/data/processed"

# Ensure jovyan owns the workspace tree.
# CHOWN_HOME handles the top-level home directory; this covers any new
# subdirectories created above that postdate the CHOWN_HOME pass.
# Uses ${NB_UID:-1000} / ${NB_GID:-100} with defaults so the script is
# safe under set -u even if the caller did not export these variables.
# The `|| true` makes the chown non-fatal when running as a non-root
# user (possible in future Jupyter Docker Stacks versions).
chown -R "${NB_UID:-1000}:${NB_GID:-100}" "${WORKSPACE}" 2>/dev/null || true

chmod 755 "${WORKSPACE}"

echo "[10-init-workspace] workspace ready: ${WORKSPACE}"