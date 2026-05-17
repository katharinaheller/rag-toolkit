from __future__ import annotations

from typing import Any, List, Mapping, Union


def extract_texts(results: List[Union[str, Mapping[str, Any]]]) -> List[str]:
    """Extract text strings from a heterogeneous list of retrieval results."""
    texts: List[str] = []
    for item in results:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            texts.append(item["text"])
    return texts


def truncate_text(text: str, max_chars: int) -> str:
    """Hard-truncate a string to at most max_chars characters."""
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0; got {max_chars}.")
    return text[:max_chars]


def format_result_summary(result: Any) -> str:
    """Compact human-readable summary of a GenerationResult."""
    answer_preview = (result.answer[:120] + "…") if len(result.answer) > 120 else result.answer
    lines = [
        f"Strategy  : {result.strategy}",
        f"Template  : {result.template_name} v{result.template_version}",
        f"Model     : {result.model}",
        f"Latency   : {result.latency_ms:.2f} ms",
        f"Chars     : prompt={result.prompt_chars}  context={result.context_chars}",
        f"Success   : {result.success}",
    ]
    if result.error:
        lines.append(f"Error     : {result.error}")
    lines.append(f"Answer    : {answer_preview}")
    return "\n".join(lines)
