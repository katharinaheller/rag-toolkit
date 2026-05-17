"""Validation tests for the immutable GenerationConfig."""

from __future__ import annotations

import pytest

from rag.generation.config import GenerationConfig


def test_minimum_valid_config() -> None:
    cfg = GenerationConfig(model_name="mistral")
    assert cfg.model_name == "mistral"
    assert cfg.url.endswith("/api/generate")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_rejects_blank_model_name(blank: str) -> None:
    with pytest.raises(ValueError, match="model_name"):
        GenerationConfig(model_name=blank)


def test_rejects_blank_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        GenerationConfig(model_name="m", base_url="")


def test_rejects_blank_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        GenerationConfig(model_name="m", endpoint="")


@pytest.mark.parametrize("t", [-0.1, -1.0, 2.01, 3.0])
def test_rejects_out_of_range_temperature(t: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_name="m", temperature=t)


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 2.0])
def test_accepts_in_range_temperature(t: float) -> None:
    GenerationConfig(model_name="m", temperature=t)


@pytest.mark.parametrize("max_tokens", [0, -1, -100])
def test_rejects_non_positive_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        GenerationConfig(model_name="m", max_tokens=max_tokens)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.1, 2.0])
def test_rejects_out_of_range_top_p(p: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_name="m", top_p=p)


def test_rejects_zero_repeat_penalty() -> None:
    with pytest.raises(ValueError, match="repeat_penalty"):
        GenerationConfig(model_name="m", repeat_penalty=0.0)


@pytest.mark.parametrize("t", [0.0, -1.0])
def test_rejects_non_positive_timeout(t: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        GenerationConfig(model_name="m", timeout=t)


def test_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        GenerationConfig(model_name="m", max_retries=-1)


def test_rejects_negative_retry_delay() -> None:
    with pytest.raises(ValueError, match="retry_delay"):
        GenerationConfig(model_name="m", retry_delay=-0.1)


def test_rejects_non_positive_context_chars() -> None:
    with pytest.raises(ValueError, match="max_context_chars"):
        GenerationConfig(model_name="m", max_context_chars=0)


def test_url_strips_trailing_slash_from_base() -> None:
    cfg = GenerationConfig(model_name="m", base_url="http://x:11434/")
    assert cfg.url == "http://x:11434/api/generate"


def test_ollama_options_includes_required_fields() -> None:
    cfg = GenerationConfig(model_name="m", temperature=0.5, max_tokens=256, seed=99)
    opts = cfg.ollama_options
    assert opts["temperature"] == 0.5
    assert opts["num_predict"] == 256
    assert opts["seed"] == 99
    assert "top_p" in opts and "repeat_penalty" in opts


def test_config_is_frozen() -> None:
    cfg = GenerationConfig(model_name="m")
    with pytest.raises(Exception):
        cfg.model_name = "other"  # type: ignore[misc]
