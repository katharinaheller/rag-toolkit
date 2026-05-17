from pathlib import Path
from typing import Optional

from rag.logging.logger import get_logger

logger = get_logger(__name__)

_ENCODINGS = ("utf-8", "latin-1")


def is_binary(path: Path) -> bool:
    """Return True if the file contains a null byte in its first 1 KiB."""
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(1024)
    except Exception as exc:
        logger.info("text_reader.binary_probe_failed", file=str(path), error=str(exc))
        return True


def read_text(path: Path) -> Optional[str]:
    """Read path as text via UTF-8 then Latin-1. Returns None on all failures."""
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.info("text_reader.read_failed", file=str(path), encoding=encoding, error=str(exc))
            return None

    logger.info("text_reader.all_encodings_failed", file=str(path))
    return None
