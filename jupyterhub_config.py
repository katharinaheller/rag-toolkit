"""
JupyterHub configuration — RAG Toolkit

Design notes
────────────
Self-inspection for bind-mount paths
    DockerSpawner creates notebook containers by calling the Docker daemon
    directly.  It must supply HOST-SIDE paths as bind mount sources.

    Those paths vary by OS (Linux /home/…, macOS /Users/…, Windows C:\…)
    and by where the repo was cloned.  We cannot hardcode them.

    Solution: ask Docker what host paths are already mounted into THIS hub
    container.  The hub container has ./notebooks and ./data bind-mounted
    at /hub-mounts/notebooks and /hub-mounts/data by Compose.  The Docker
    inspect API returns the host-side source for each mount.

    Because DockerSpawner and the hub share the same Docker daemon,
    passing those paths back to the daemon for notebook container mounts
    is always correct — even on Docker Desktop, where the daemon handles
    the host↔VM translation layer internally.

pull_policy = "never"
    rag-notebook:0.2.0 is a local image built by the notebook-builder
    Compose service.  It does not exist on Docker Hub or any registry.
    DockerSpawner must never attempt to pull it.
"""

import os
import sys
import socket

import docker

# ── get_config() is injected by JupyterHub's config machinery ───────────
c = get_config()  # noqa: F821


# ════════════════════════════════════════════════════════════════════════
# DockerSpawner — core settings
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"

c.DockerSpawner.image = os.environ.get(
    "DOCKER_NOTEBOOK_IMAGE",
    "rag-notebook:0.2.0",
)

# ── CRITICAL ──────────────────────────────────────────────────────────
# Never pull from a registry.  The image is built locally by the
# notebook-builder Compose service before hub ever starts.
# Default is "always", which fails with:
#   pull access denied for rag-notebook, repository does not exist
# ────────────────────────────────────────────────────────────────────
c.DockerSpawner.pull_policy = "never"

c.DockerSpawner.network_name = os.environ.get(
    "DOCKER_NETWORK_NAME",
    "rag-toolkit_backend",
)

c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True

c.DockerSpawner.notebook_dir = "/home/jovyan"

c.Spawner.default_url = "/lab/tree/shared/notebooks"


# ════════════════════════════════════════════════════════════════════════
# Resolve host-side bind-mount paths via Docker self-inspection
# ════════════════════════════════════════════════════════════════════════

def _connect_docker() -> docker.APIClient:
    """Return a Docker API client connected to the local daemon socket.
    Raises with an actionable message if the socket is not accessible."""
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
    """Inspect the hub container.  In a Docker container socket.gethostname()
    returns the container short ID, which the Docker daemon accepts."""
    container_id = socket.gethostname()
    try:
        return client.inspect_container(container_id)
    except docker.errors.NotFound:
        print(
            "\n[jupyterhub_config] FATAL: Cannot inspect hub container "
            f"'{container_id}'.\n"
            "  This usually means the Docker daemon cannot find a running "
            "container with that ID.\n"
            "  Verify that container_name: hub in docker-compose.yml matches "
            "the actual running container.\n",
            file=sys.stderr,
        )
        raise


def _extract_bind_mounts(info: dict) -> dict[str, str]:
    """Return {container_destination: host_source} for all bind mounts."""
    return {
        m["Destination"]: m["Source"]
        for m in info["Mounts"]
        if m["Type"] == "bind"
    }


def _require_mount(destination: str, mounts: dict[str, str]) -> str:
    """Return the host source path for a required mount destination.
    Raises RuntimeError with an actionable message if missing."""
    source = mounts.get(destination)
    if source:
        return source
    lines = ["".join(f"    {d!r:40s} -> {s}\n" for d, s in mounts.items())]
    raise RuntimeError(
        f"\n[jupyterhub_config] FATAL: Required bind mount not found.\n"
        f"  expected destination : {destination!r}\n"
        f"  available mounts     :\n{''.join(lines)}"
        "\n  Ensure docker-compose.yml includes:\n"
        "    volumes:\n"
        "      - ./notebooks:/hub-mounts/notebooks:ro\n"
        "      - ./data:/hub-mounts/data:ro\n"
    )


_docker_client  = _connect_docker()
_hub_info       = _inspect_self(_docker_client)
_bind_mounts    = _extract_bind_mounts(_hub_info)

notebooks_host_path = _require_mount("/hub-mounts/notebooks", _bind_mounts)
data_host_path      = _require_mount("/hub-mounts/data",      _bind_mounts)


# ════════════════════════════════════════════════════════════════════════
# DockerSpawner — volume mounts for notebook containers
# ════════════════════════════════════════════════════════════════════════

c.DockerSpawner.volumes = {

    # Per-user persistent workspace — named volume (rw).
    # Docker seeds the volume from the image's /home/jovyan/workspace on
    # first use, so workspace/notebooks and workspace/data/* exist
    # immediately without any manual setup.
    "jupyterhub-work-{username}": {
        "bind": "/home/jovyan/workspace",
        "mode": "rw",
    },

    # Shared notebooks — host bind mount (ro).
    # Source path resolved dynamically above: OS-independent.
    notebooks_host_path: {
        "bind": "/home/jovyan/shared/notebooks",
        "mode": "ro",
    },

    # Shared data — host bind mount (ro).
    data_host_path: {
        "bind": "/home/jovyan/shared/data",
        "mode": "ro",
    },

    # HuggingFace model cache — named volume shared across all users (rw).
    "hf-cache": {
        "bind": "/srv/hf-cache",
        "mode": "rw",
    },
}


# ════════════════════════════════════════════════════════════════════════
# Spawner environment injected into every notebook container
# ════════════════════════════════════════════════════════════════════════

c.Spawner.environment = {

    # ── Jupyter Docker Stacks startup ─────────────────────────────────
    "DOCKER_STACKS_JUPYTER_CMD": "lab",
    "JUPYTER_ENABLE_LAB":        "yes",
    # JUPYTER_ENV_VARS_TO_UNSET is intentionally omitted.
    # Setting it to "" causes `unset ""` (bad variable name) in bash-based
    # startup hooks.  Omitting it entirely means "unset nothing".
    "GRANT_SUDO":                "no",
    "RESTARTABLE":               "yes",
    "NB_UID":                    "1000",
    "NB_GID":                    "100",
    "CHOWN_HOME":                "yes",
    "CHOWN_HOME_OPTS":           "-R",

    # ── HuggingFace cache ─────────────────────────────────────────────
    # Must match the /srv/hf-cache volume mount above.
    "HF_HOME":               "/srv/hf-cache",
    "HUGGINGFACE_HUB_CACHE": "/srv/hf-cache",
    "TRANSFORMERS_CACHE":    "/srv/hf-cache",

    # ── Convenience paths for user notebooks ─────────────────────────
    "RAG_SHARED_NOTEBOOKS": "/home/jovyan/shared/notebooks",
    "RAG_SHARED_DATA":      "/home/jovyan/shared/data",
    "RAG_WORKSPACE":        "/home/jovyan/workspace",
}


# ════════════════════════════════════════════════════════════════════════
# Hub networking
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.bind_url       = "http://0.0.0.0:8000"
c.JupyterHub.hub_ip         = "0.0.0.0"
c.JupyterHub.hub_connect_ip = "hub"   # DNS name of the hub container


# ════════════════════════════════════════════════════════════════════════
# Persistence
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.cookie_secret_file = "/data/jupyterhub_cookie_secret"
c.JupyterHub.db_url             = "sqlite:////data/jupyterhub.sqlite"


# ════════════════════════════════════════════════════════════════════════
# Authentication
# ════════════════════════════════════════════════════════════════════════

c.JupyterHub.authenticator_class  = "nativeauthenticator.NativeAuthenticator"
c.Authenticator.allow_all         = True
c.NativeAuthenticator.open_signup = True

_admin = os.environ.get("JUPYTERHUB_ADMIN", "")
if _admin:
    c.Authenticator.admin_users = {_admin}