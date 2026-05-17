from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Cleaner(Protocol):
    """Contract: normalize raw text, or return None if result is empty."""

    def clean(self, text: str) -> Optional[str]:
        ...


class DefaultCleaner:
    """Minimal, deterministic text normalizer.

    Steps: strip BOM, normalize line endings to LF, strip trailing horizontal
    whitespace per line, remove trailing blank lines, return None when blank.
    """

    def clean(self, text: str) -> Optional[str]:
        if text.startswith("\ufeff"):
            text = text[1:]

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip(" \t") for line in text.split("\n")]

        while lines and lines[-1] == "":
            lines.pop()

        cleaned = "\n".join(lines)

        if not cleaned.strip():
            return None

        return cleaned
