"""Hardware, CUDA and Torch metadata collection.

Every benchmark records the host environment so results are reproducible and
comparable across machines. All imports are guarded — the module never raises
on a CPU-only system and never crashes when ``pynvml`` or ``torch`` are
missing.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Torch / CUDA probing (fail-soft) ────────────────────────────────────────
def _probe_torch() -> Dict[str, Any]:
    """Return a metadata dict describing the local torch + CUDA install."""
    info: Dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "default_device": "cpu",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = f"torch not installed: {exc}"
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", "unknown")

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    info["cuda_available"] = cuda_ok
    if not cuda_ok:
        return info

    try:
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        info["cuda_version"] = None
    try:
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
    except Exception:
        info["cudnn_version"] = None

    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    info["device_count"] = n
    info["default_device"] = "cuda:0" if n > 0 else "cpu"

    devices: List[Dict[str, Any]] = []
    for i in range(n):
        device: Dict[str, Any] = {"index": i}
        try:
            device["name"] = torch.cuda.get_device_name(i)
        except Exception:
            device["name"] = None
        try:
            props = torch.cuda.get_device_properties(i)
            device.update({
                "total_memory_mb": round(float(props.total_memory) / (1024 * 1024), 1),
                "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                "major": int(getattr(props, "major", 0)),
                "minor": int(getattr(props, "minor", 0)),
            })
        except Exception:
            pass
        devices.append(device)
    info["devices"] = devices
    return info


def _probe_pynvml() -> Dict[str, Any]:
    """Return whatever NVML can tell us about GPU 0."""
    info: Dict[str, Any] = {"pynvml_available": False}
    try:
        import pynvml  # type: ignore
    except Exception as exc:
        info["error"] = f"pynvml not installed: {exc}"
        return info
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        info["error"] = f"nvmlInit failed: {exc}"
        return info
    info["pynvml_available"] = True
    try:
        n = int(pynvml.nvmlDeviceGetCount())
    except Exception:
        n = 0
    info["device_count"] = n
    devices: List[Dict[str, Any]] = []
    for i in range(n):
        d: Dict[str, Any] = {"index": i}
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            d["name"] = name
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                d["total_memory_mb"] = round(float(mem.total) / (1024 * 1024), 1)
            except Exception:
                pass
            try:
                d["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(d["driver_version"], bytes):
                    d["driver_version"] = d["driver_version"].decode("utf-8", errors="replace")
            except Exception:
                pass
        except Exception as exc:
            d["error"] = str(exc)
        devices.append(d)
    info["devices"] = devices
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return info


def _probe_cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "system": platform.system(),
        "release": platform.release(),
        "logical_cores": os.cpu_count() or 0,
    }
    try:
        import psutil  # type: ignore
        info["physical_cores"] = psutil.cpu_count(logical=False) or 0
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(float(vm.total) / (1024 ** 3), 2)
        info["available_ram_gb"] = round(float(vm.available) / (1024 ** 3), 2)
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                info["cpu_freq_mhz"] = round(float(freq.current), 1)
        except Exception:
            pass
    except Exception:
        info["physical_cores"] = None
        info["total_ram_gb"] = None
        info["available_ram_gb"] = None
    return info


@dataclass
class HardwareMetadata:
    """Snapshot of the host environment at benchmark start."""

    hostname: str = ""
    python_version: str = ""
    platform_str: str = ""
    cpu: Dict[str, Any] = field(default_factory=dict)
    torch: Dict[str, Any] = field(default_factory=dict)
    nvml: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def cuda_available(self) -> bool:
        return bool(self.torch.get("cuda_available"))

    @property
    def gpu_count(self) -> int:
        return int(self.torch.get("device_count", 0) or 0)

    @property
    def primary_gpu_name(self) -> Optional[str]:
        devs = self.torch.get("devices") or []
        if devs and devs[0].get("name"):
            return devs[0]["name"]
        nvml_devs = self.nvml.get("devices") or []
        if nvml_devs and nvml_devs[0].get("name"):
            return nvml_devs[0]["name"]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "python_version": self.python_version,
            "platform": self.platform_str,
            "cpu": self.cpu,
            "torch": self.torch,
            "nvml": self.nvml,
            "env": self.env,
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "primary_gpu_name": self.primary_gpu_name,
        }


def collect_hardware_metadata() -> HardwareMetadata:
    """Capture full hardware/Torch/CUDA snapshot. Never raises."""
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    )
    env = {k: os.environ.get(k, "") for k in env_keys}

    return HardwareMetadata(
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
        platform_str=platform.platform(),
        cpu=_probe_cpu(),
        torch=_probe_torch(),
        nvml=_probe_pynvml(),
        env=env,
    )


def cuda_is_available() -> bool:
    """Light check used by benchmark guards. Never raises."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_seed_all(seed: int) -> None:
    """Set seeds for torch / cuda / random / numpy where importable."""
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)  # noqa: NPY002
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def reset_gpu_peak_memory(device_index: int = 0) -> None:
    """Reset torch's per-device peak memory counter. Safe on CPU-only hosts."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_index)
    except Exception:
        pass


def gpu_peak_memory_mb(device_index: int = 0) -> Optional[float]:
    """Peak allocated GPU memory in MiB since last reset, or None on CPU."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated(device_index)) / (1024 * 1024)
    except Exception:
        return None


def gpu_allocated_memory_mb(device_index: int = 0) -> Optional[float]:
    """Currently allocated GPU memory in MiB."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated(device_index)) / (1024 * 1024)
    except Exception:
        return None


def gpu_synchronize(device_index: int = 0) -> None:
    """Block until the given device finishes pending kernels."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_index)
    except Exception:
        pass


def empty_gpu_cache() -> None:
    """Release cached blocks held by the torch allocator. Best-effort."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
