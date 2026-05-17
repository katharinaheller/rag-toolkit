"""Lightweight logging configuration and run-id minting.

Pure stdlib so the framework runs anywhere with no extra dependency.
"""

from __future__ import annotations

import datetime as _dt
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional


_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", logfile: Optional[Path] = None) -> None:
    """Configure root logging once. Idempotent."""
    root = logging.getLogger()
    # Avoid duplicate handlers when called twice (Jupyter cells, re-runs).
    if getattr(root, "_experiments_configured", False):
        if logfile is not None and not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == logfile
            for h in root.handlers
        ):
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_DEFAULT_FMT))
            root.addHandler(fh)
        return

    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(logging.Formatter(_DEFAULT_FMT))
    root.addHandler(stream)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_DEFAULT_FMT))
        root.addHandler(fh)

    root._experiments_configured = True  # type: ignore[attr-defined]


def new_run_id(prefix: str = "run") -> str:
    """Mint a timestamped, unique run identifier."""
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
