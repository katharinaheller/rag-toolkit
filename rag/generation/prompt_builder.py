from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt design specification.

    Increment `version` when wording changes affect model behavior so that
    template_version is captured correctly in GenerationResult.
    """

    name: str
    version: str
    system_prompt: str
    user_template: str
    separator: str = "\n\n---\n\n"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PromptTemplate.name must be a non-empty string.")
        if not self.version or not self.version.strip():
            raise ValueError("PromptTemplate.version must be a non-empty string.")
        if "{context}" not in self.user_template:
            raise ValueError("PromptTemplate.user_template must contain {context}.")
        if "{query}" not in self.user_template:
            raise ValueError("PromptTemplate.user_template must contain {query}.")


STRICT_RAG_TEMPLATE = PromptTemplate(
    name="strict_rag",
    version="1.0.0",
    system_prompt=(
        "You are a precise, factual assistant.\n"
        "Answer ONLY using the information provided in the context below.\n"
        "If the answer cannot be found in the context, respond with exactly:\n"
        "\"I don't know.\"\n"
        "Do not add information from your training data."
    ),
    user_template=(
        "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    ),
)

COT_TEMPLATE = PromptTemplate(
    name="chain_of_thought",
    version="1.0.0",
    system_prompt=(
        "You are a careful reasoning assistant.\n"
        "Use the provided context to answer the question step by step.\n"
        "For each reasoning step, explicitly cite the relevant part of the context.\n"
        "Conclude with a concise final answer.\n"
        "Do not introduce facts not present in the context."
    ),
    user_template=(
        "Context:\n{context}\n\nQuestion:\n{query}\n\nStep-by-step reasoning:\n"
    ),
)

FEW_SHOT_TEMPLATE = PromptTemplate(
    name="few_shot_rag",
    version="1.0.0",
    system_prompt=(
        "You are a precise assistant that answers questions using only the given context.\n"
        "Study the examples below, then answer the final question in the same style."
    ),
    user_template=(
        "### Example 1\n"
        "Context:\n"
        "Retrieval-Augmented Generation (RAG) combines a retrieval step with a "
        "generative model to ground answers in a document corpus.\n\n"
        "Question:\nWhat does RAG stand for?\n\n"
        "Answer:\nRAG stands for Retrieval-Augmented Generation.\n\n"
        "---\n\n"
        "### Example 2\n"
        "Context:\nMistral 7B is a 7-billion-parameter language model released by Mistral AI.\n\n"
        "Question:\nHow many parameters does Mistral 7B have?\n\n"
        "Answer:\nMistral 7B has 7 billion parameters.\n\n"
        "---\n\n"
        "### Your turn\n"
        "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    ),
)

REFINE_INITIAL_TEMPLATE = PromptTemplate(
    name="refine_initial",
    version="1.0.0",
    system_prompt=(
        "You are a factual assistant.\n"
        "Produce a concise initial answer using only the context provided.\n"
        "If the context is insufficient, say so explicitly."
    ),
    user_template="Context:\n{context}\n\nQuestion:\n{query}\n\nInitial answer:",
)

# Refine-update reuses {context} for the existing answer and {query} for the new chunk.
REFINE_UPDATE_TEMPLATE = PromptTemplate(
    name="refine_update",
    version="1.0.0",
    system_prompt=(
        "You are a factual assistant.\n"
        "You have an existing answer and additional context.\n"
        "Refine the existing answer if the new context adds relevant information.\n"
        "If not, return the existing answer unchanged.\n"
        "Do not introduce information absent from either source."
    ),
    user_template=(
        "Existing answer:\n{context}\n\nNew context:\n{query}\n\nRefined answer:"
    ),
)


class PromptBuilder:
    """Stateless builder that assembles final prompt strings from a template."""

    def __init__(self, template: PromptTemplate) -> None:
        self._template = template

    @property
    def template(self) -> PromptTemplate:
        return self._template

    def build(self, query: str, context_chunks: List[str]) -> str:
        """Assemble the final prompt string."""
        context_block = self._template.separator.join(context_chunks)
        user_section = self._template.user_template.format(context=context_block, query=query)
        return f"{self._template.system_prompt}\n\n{user_section}"
