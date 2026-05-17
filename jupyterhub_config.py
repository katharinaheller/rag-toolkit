"""
JupyterHub configuration — RAG Toolkit
"""

from __future__ import annotations

import os
import socket
import sys
from typing import TYPE_CHECKING, Any

import docker


# Define a static-analysis fallback for editors like Pylance.
# JupyterHub injects the real get_config() function at runtime.
if TYPE_CHECKING:

    def get_config() -> Any:
        ...


# JupyterHub runtime configuration object.
c = get_config()


# DockerSpawner core settings
c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"

c.DockerSpawner.image = os.environ.get(
    "DOCKER_NOTEBOOK_IMAGE",
    "rag-notebook:0.2.0",
)

c.DockerSpawner.pull_policy = "never"

c.DockerSpawner.network_name = os.environ.get(
    "DOCKER_NETWORK_NAME",
    "rag-toolkit_backend",
)

c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.remove = True

c.DockerSpawner.notebook_dir = "/home/jovyan"

c.Spawner.default_url = "/lab/tree/shared/notebooks"


# Create a Docker API client connected to the local daemon socket.
def _connect_docker() -> docker.APIClient:

    socket_url = "unix:///var/run/docker.sock"

    try:
        return docker.APIClient(base_url=socket_url)

    except Exception as exc:

        print(
            "\n[jupyterhub_config] FATAL: Cannot connect to Docker daemon.\n"
            f"  socket : {socket_url}\n"
            f"  error  : {exc}\n",
            file=sys.stderr,
        )

        raise


# Inspect the currently running hub container.
def _inspect_self(client: docker.APIClient) -> dict:

    container_id = socket.gethostname()

    try:
        return client.inspect_container(container_id)

    except docker.errors.NotFound:

        print(
            "\n[jupyterhub_config] FATAL: Cannot inspect hub container "
            f"'{container_id}'.\n",
            file=sys.stderr,
        )

        raise


# Extract all bind mounts from the container inspection result.
def _extract_bind_mounts(info: dict) -> dict[str, str]:

    return {
        mount["Destination"]: mount["Source"]
        for mount in info["Mounts"]
        if mount["Type"] == "bind"
    }


# Resolve a required bind mount destination to its host path.
def _require_mount(
    destination: str,
    mounts: dict[str, str],
) -> str:

    source = mounts.get(destination)

    if source:
        return source

    available = "".join(
        f"    {dst!r:40s} -> {src}\n"
        for dst, src in mounts.items()
    )

    raise RuntimeError(
        "\n[jupyterhub_config] FATAL: Required bind mount not found.\n"
        f"  expected destination : {destination!r}\n"
        f"  available mounts     :\n{available}"
    )


_docker_client = _connect_docker()
_hub_info = _inspect_self(_docker_client)
_bind_mounts = _extract_bind_mounts(_hub_info)

notebooks_host_path = _require_mount(
    "/hub-mounts/notebooks",
    _bind_mounts,
)

data_host_path = _require_mount(
    "/hub-mounts/data",
    _bind_mounts,
)


# DockerSpawner volume mounts
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


# Environment variables injected into notebook containers
c.Spawner.environment = {

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

    "RAG_SHARED_NOTEBOOKS": "/home/jovyan/shared/notebooks",
    "RAG_SHARED_DATA": "/home/jovyan/shared/data",
    "RAG_WORKSPACE": "/home/jovyan/workspace",
}


# Hub networking
c.JupyterHub.bind_url = "http://0.0.0.0:8000"
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.hub_connect_ip = "hub"


# Persistence
c.JupyterHub.cookie_secret_file = "/data/jupyterhub_cookie_secret"

c.JupyterHub.db_url = (
    "sqlite:////data/jupyterhub.sqlite"
)


# Authentication
c.JupyterHub.authenticator_class = (
    "nativeauthenticator.NativeAuthenticator"
)

c.Authenticator.allow_all = True
c.NativeAuthenticator.open_signup = True


_admin = os.environ.get("JUPYTERHUB_ADMIN", "")

if _admin:
    c.Authenticator.admin_users = {_admin}