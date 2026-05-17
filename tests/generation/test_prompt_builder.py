"""Tests for PromptTemplate and PromptBuilder."""

from __future__ import annotations

import pytest

from rag.generation.prompt_builder import (
    COT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    REFINE_INITIAL_TEMPLATE,
    REFINE_UPDATE_TEMPLATE,
    STRICT_RAG_TEMPLATE,
    PromptBuilder,
    PromptTemplate,
)


class TestPromptTemplateValidation:
    def test_minimum_valid_template(self) -> None:
        t = PromptTemplate(name="t", version="1.0", system_prompt="x",
                           user_template="ctx={context} q={query}")
        assert t.name == "t"

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_blank_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="name"):
            PromptTemplate(name=name, version="1.0", system_prompt="x",
                           user_template="ctx={context} q={query}")

    @pytest.mark.parametrize("v", ["", " "])
    def test_rejects_blank_version(self, v: str) -> None:
        with pytest.raises(ValueError, match="version"):
            PromptTemplate(name="n", version=v, system_prompt="x",
                           user_template="ctx={context} q={query}")

    def test_user_template_must_include_context(self) -> None:
        with pytest.raises(ValueError, match="\\{context\\}"):
            PromptTemplate(name="n", version="1", system_prompt="x",
                           user_template="q={query}")

    def test_user_template_must_include_query(self) -> None:
        with pytest.raises(ValueError, match="\\{query\\}"):
            PromptTemplate(name="n", version="1", system_prompt="x",
                           user_template="ctx={context}")

    def test_is_frozen(self) -> None:
        t = STRICT_RAG_TEMPLATE
        with pytest.raises(Exception):
            t.name = "other"  # type: ignore[misc]


class TestBuiltInTemplates:
    @pytest.mark.parametrize("template", [
        STRICT_RAG_TEMPLATE, COT_TEMPLATE, FEW_SHOT_TEMPLATE,
        REFINE_INITIAL_TEMPLATE, REFINE_UPDATE_TEMPLATE,
    ])
    def test_template_has_required_fields(self, template: PromptTemplate) -> None:
        assert template.name
        assert template.version
        assert "{context}" in template.user_template
        assert "{query}" in template.user_template


class TestPromptBuilder:
    def test_builds_full_prompt(self) -> None:
        b = PromptBuilder(STRICT_RAG_TEMPLATE)
        prompt = b.build("What is X?", ["chunk one", "chunk two"])
        assert "chunk one" in prompt
        assert "chunk two" in prompt
        assert "What is X?" in prompt
        assert STRICT_RAG_TEMPLATE.system_prompt.split("\n")[0] in prompt

    def test_separator_inserted_between_chunks(self) -> None:
        b = PromptBuilder(STRICT_RAG_TEMPLATE)
        prompt = b.build("q", ["A", "B"])
        assert STRICT_RAG_TEMPLATE.separator in prompt
        # Order preserved
        assert prompt.index("A") < prompt.index("B")

    def test_empty_context_does_not_raise(self) -> None:
        b = PromptBuilder(STRICT_RAG_TEMPLATE)
        prompt = b.build("q", [])
        assert "q" in prompt

    def test_template_property(self) -> None:
        b = PromptBuilder(STRICT_RAG_TEMPLATE)
        assert b.template is STRICT_RAG_TEMPLATE

    def test_query_with_braces_does_not_break_format(self) -> None:
        """Build should embed query literally even if it contains brace-like text."""
        b = PromptBuilder(STRICT_RAG_TEMPLATE)
        prompt = b.build("Use {placeholder} carefully", ["c"])
        assert "{placeholder}" in prompt
