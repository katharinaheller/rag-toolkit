"""
JupyterHub configuration — RAG Toolkit

Design notes
────────────
Self-inspection for bind-mount paths
    DockerSpawner creates notebook containers by calling the Docker daemon
    directly and must provide HOST-SIDE paths as bind mount sources. Those
    paths vary by OS and clone location, so the hub inspects itself through
    the Docker API and reuses the resolved host paths.

Student isolation for /opt/experiments
    The experiments tree is mounted into the HUB only (for path resolution).
    It is NOT part of c.DockerSpawner.volumes, so it is never attached to
    student notebooks. An admin-only pre_spawn_hook forwards it read-only to
    /opt/experiments for admin users exclusively. The dedicated
    benchmark-runner container remains the recommended GPU execution path.

GPU enablement
    DockerSpawner must explicitly request GPU devices via Docker's
    DeviceRequest API; notebooks otherwise launch without CUDA.

HuggingFace authentication forwarding
    The hub container receives HF_TOKEN / HUGGINGFACE_HUB_TOKEN from .env via
    docker-compose interpolation. This config reads them from the hub's own
    environment and injects BOTH names into every spawned notebook container
    so huggingface_hub, transformers and sentence-transformers all detect the
    token automatically and can pull gated repos. The token is runtime-only:
    it is never written to an image, a layer, or any source file.
"""

import os
import socket
import sys
from typing import TYPE_CHECKING, Any, Optional

import docker


# Static-analysis fallback for editors like Pylance.
if TYPE_CHECKING:

    def get_config() -> Any:
        ...


c = get_config()


# ════════════════════════════════════════════════════════════════════════
# DockerSpawner — core settings
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"

_NOTEBOOK_IMAGE = os.environ.get("DOCKER_NOTEBOOK_IMAGE", "rag-notebook:0.2.0")

c.DockerSpawner.image = _NOTEBOOK_IMAGE

c.DockerSpawner.pull_policy = "never"

c.DockerSpawner.network_name = os.environ.get(
    "DOCKER_NETWORK_NAME",
    "rag-toolkit_backend",
)

c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True


# ───────────────────────────────────────────────────────────────────────
# CRITICAL GPU FIX — explicit DeviceRequest for notebook containers.
# ───────────────────────────────────────────────────────────────────────
if _NOTEBOOK_IMAGE.endswith("-gpu:0.2.0"):
    c.DockerSpawner.extra_host_config = {
        "device_requests": [
            docker.types.DeviceRequest(
                count=-1,
                capabilities=[["gpu"]],
            )
        ]
    }

c.DockerSpawner.notebook_dir = "/home/jovyan"

c.Spawner.default_url = "/lab/tree/shared/notebooks"


# ════════════════════════════════════════════════════════════════════════
# HuggingFace token — read from the hub environment (set via .env)
# ════════════════════════════════════════════════════════════════════════

def _resolve_hf_token() -> str:
    """Return the HF token from the hub environment, '' when unset/blank."""
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


_HF_TOKEN = _resolve_hf_token()

if _HF_TOKEN:
    print(
        "[jupyterhub_config] HuggingFace token detected; forwarding "
        "HF_TOKEN + HUGGINGFACE_HUB_TOKEN to spawned containers.",
        file=sys.stderr,
    )
else:
    print(
        "[jupyterhub_config] NOTE: no HF token in environment. Gated model "
        "downloads will fail with 401. Set HF_TOKEN in .env and recreate the "
        "hub container.",
        file=sys.stderr,
    )


# ════════════════════════════════════════════════════════════════════════
# Resolve host-side bind-mount paths via Docker self-inspection
# ════════════════════════════════════════════════════════════════════════

def _connect_docker() -> docker.APIClient:
    socket_url = "unix:///var/run/docker.sock"
    try:
        return docker.APIClient(base_url=socket_url)
    except Exception as exc:
        print(
            "\n[jupyterhub_config] FATAL: Cannot connect to Docker daemon.\n"
            f"  socket : {socket_url}\n"
            f"  error  : {exc}\n"
            "\n  Ensure docker-compose.yml mounts the Docker socket:\n"
            "    volumes:\n"
            "      - /var/run/docker.sock:/var/run/docker.sock\n",
            file=sys.stderr,
        )
        raise


def _inspect_self(client: docker.APIClient) -> dict:
    container_id = socket.gethostname()
    try:
        return client.inspect_container(container_id)
    except docker.errors.NotFound:
        print(
            "\n[jupyterhub_config] FATAL: Cannot inspect hub container.\n"
            f"  container : {container_id}\n"
            "\n  Ensure the hub container is running and accessible.\n",
            file=sys.stderr,
        )
        raise


def _extract_bind_mounts(info: dict) -> dict:
    return {
        mount["Destination"]: mount["Source"]
        for mount in info["Mounts"]
        if mount["Type"] == "bind"
    }


def _require_mount(destination: str, mounts: dict) -> str:
    source = mounts.get(destination)
    if source:
        return source
    available_mounts = "".join(
        f"    {dest!r:40s} -> {src}\n"
        for dest, src in mounts.items()
    )
    raise RuntimeError(
        f"\n[jupyterhub_config] FATAL: Required bind mount not found.\n"
        f"  expected destination : {destination!r}\n"
        f"  available mounts     :\n{available_mounts}"
        "\n  Ensure docker-compose.yml includes:\n"
        "    volumes:\n"
        "      - ./notebooks:/hub-mounts/notebooks:ro\n"
        "      - ./data:/hub-mounts/data:ro\n"
    )


def _optional_mount(destination: str, mounts: dict) -> Optional[str]:
    source = mounts.get(destination)
    if not source:
        print(
            "[jupyterhub_config] NOTE: optional mount "
            f"{destination!r} not present; admin in-Lab experiments path "
            "disabled (use the benchmark-runner container instead).",
            file=sys.stderr,
        )
    return source


_docker_client = _connect_docker()
_hub_info = _inspect_self(_docker_client)
_bind_mounts = _extract_bind_mounts(_hub_info)

notebooks_host_path = _require_mount("/hub-mounts/notebooks", _bind_mounts)
data_host_path = _require_mount("/hub-mounts/data", _bind_mounts)
experiments_host_path = _optional_mount("/hub-mounts/experiments", _bind_mounts)


# ════════════════════════════════════════════════════════════════════════
# DockerSpawner — DEFAULT notebook container volume mounts (all users)
# ════════════════════════════════════════════════════════════════════════

c.DockerSpawner.volumes = {

    "jupyterhub-work-{username}": {
        "bind": "/home/jovyan/workspace",
        "mode": "rw",
    },

    notebooks_host_path: {
        "bind": "/home/jovyan/shared/notebooks",
        "mode": "ro",
    },

    data_host_path: {
        "bind": "/home/jovyan/shared/data",
        "mode": "ro",
    },

    "hf-cache": {
        "bind": "/srv/hf-cache",
        "mode": "rw",
    },
}


# ════════════════════════════════════════════════════════════════════════
# Environment variables injected into notebook containers (all users)
# ════════════════════════════════════════════════════════════════════════

_spawner_env = {

    "DOCKER_STACKS_JUPYTER_CMD": "lab",
    "JUPYTER_ENABLE_LAB": "yes",
    "GRANT_SUDO": "no",
    "RESTARTABLE": "yes",

    "NB_UID": "1000",
    "NB_GID": "100",
    "CHOWN_HOME": "yes",
    "CHOWN_HOME_OPTS": "-R",

    "HF_HOME": "/srv/hf-cache",
    "HUGGINGFACE_HUB_CACHE": "/srv/hf-cache",
    "TRANSFORMERS_CACHE": "/srv/hf-cache",
    "HF_HUB_DISABLE_TELEMETRY": "1",

    "RAG_SHARED_NOTEBOOKS": "/home/jovyan/shared/notebooks",
    "RAG_SHARED_DATA": "/home/jovyan/shared/data",
    "RAG_WORKSPACE": "/home/jovyan/workspace",
}

# Forward the HuggingFace token under BOTH canonical names when present.
if _HF_TOKEN:
    _spawner_env["HF_TOKEN"] = _HF_TOKEN
    _spawner_env["HUGGINGFACE_HUB_TOKEN"] = _HF_TOKEN

c.Spawner.environment = _spawner_env


# ════════════════════════════════════════════════════════════════════════
# Admin-only experiments forwarding (in-Lab convenience path)
# ════════════════════════════════════════════════════════════════════════

def _is_admin(spawner) -> bool:
    user = getattr(spawner, "user", None)
    return bool(getattr(user, "admin", False))


def admin_experiments_hook(spawner) -> None:
    """Forward experiments (ro) + EXPERIMENTS_* env to admin spawns only."""
    if experiments_host_path is None:
        return
    if not _is_admin(spawner):
        return

    volumes = dict(spawner.volumes)
    volumes[experiments_host_path] = {
        "bind": "/opt/experiments",
        "mode": "ro",
    }
    spawner.volumes = volumes

    out_root = "/home/jovyan/workspace/experiment-outputs"
    env = dict(spawner.environment)
    env.update({
        "PYTHONPATH": "/opt",
        "PYTHONDONTWRITEBYTECODE": "1",
        "EXPERIMENTS_OUTPUT_ROOT": out_root,
        "EXPERIMENTS_CACHE_ROOT": out_root + "/indexes",
        "EXPERIMENTS_DATA_ROOT": "/home/jovyan/shared/data/documents",
        "EXPERIMENTS_ENABLE_GPU_BENCHMARKS": "1",
        "EXPERIMENTS_ENABLE_GENERATION": "0",
        "EXPERIMENTS_OLLAMA_URL": "http://ollama:11434",
        "EXPERIMENTS_OLLAMA_MODEL": "mistral",
        "EXPERIMENTS_GPU_INDEX": "0",
        "EXPERIMENTS_LOG_LEVEL": "INFO",
        "MPLCONFIGDIR": "/tmp/mpl",
    })
    # Re-assert the token for admin spawns (in case it was unset above).
    if _HF_TOKEN:
        env["HF_TOKEN"] = _HF_TOKEN
        env["HUGGINGFACE_HUB_TOKEN"] = _HF_TOKEN
    spawner.environment = env


c.Spawner.pre_spawn_hook = admin_experiments_hook


# ════════════════════════════════════════════════════════════════════════
# Hub networking
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.bind_url = "http://0.0.0.0:8000"
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.hub_connect_ip = "hub"


# ════════════════════════════════════════════════════════════════════════
# Persistence
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.cookie_secret_file = "/data/jupyterhub_cookie_secret"
c.JupyterHub.db_url = "sqlite:////data/jupyterhub.sqlite"


# ════════════════════════════════════════════════════════════════════════
# Authentication
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.authenticator_class = "nativeauthenticator.NativeAuthenticator"

c.Authenticator.allow_all = True
c.NativeAuthenticator.open_signup = True

_admin = os.environ.get("JUPYTERHUB_ADMIN", "")

if _admin:
    c.Authenticator.admin_users = {_admin}
