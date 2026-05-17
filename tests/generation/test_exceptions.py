"""Tests for the generation exception hierarchy."""

from __future__ import annotations

import pytest

from rag.generation.exceptions import (
    GenerationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)


@pytest.mark.parametrize("cls", [LLMConnectionError, LLMTimeoutError, LLMResponseError])
def test_subclasses_inherit_generation_error(cls) -> None:
    assert issubclass(cls, GenerationError)


def test_message_attribute_is_set() -> None:
    err = LLMConnectionError("network down")
    assert err.message == "network down"
    assert "network down" in str(err)


def test_repr_includes_class_and_message() -> None:
    err = LLMTimeoutError("timed out after 30s")
    rep = repr(err)
    assert "LLMTimeoutError" in rep
    assert "timed out" in rep
