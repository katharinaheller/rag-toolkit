"""Tests for GenerationResult dataclass behaviour."""

from __future__ import annotations

import pytest

from rag.generation.models import GenerationResult


def _result(answer: str = "hello", error: str | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer, prompt="p", model="m", strategy="s",
        template_name="n", template_version="v", error=error,
    )


def test_is_frozen() -> None:
    r = _result()
    with pytest.raises(Exception):
        r.answer = "different"  # type: ignore[misc]


def test_success_true_for_non_empty_answer_with_no_error() -> None:
    assert _result(answer="hi").success is True


def test_success_false_when_answer_empty() -> None:
    assert _result(answer="").success is False


def test_success_false_when_error_present() -> None:
    assert _result(answer="hi", error="boom").success is False


def test_to_dict_round_trip_fields_present() -> None:
    d = _result().to_dict()
    for key in ("answer", "prompt", "model", "strategy", "template_name",
                "template_version", "latency_ms", "prompt_chars", "context_chars",
                "inference_parameters", "timestamp", "error", "success"):
        assert key in d
