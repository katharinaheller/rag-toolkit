"""rag.generation - local RAG generation layer."""

from rag.generation.base import BaseGenerationStrategy
from rag.generation.client import OllamaClient
from rag.generation.config import GenerationConfig
from rag.generation.context import ContextPreparer, sanitize_context
from rag.generation.exceptions import (
    GenerationError,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from rag.generation.models import GenerationResult
from rag.generation.prompt_builder import (
    COT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    REFINE_INITIAL_TEMPLATE,
    REFINE_UPDATE_TEMPLATE,
    STRICT_RAG_TEMPLATE,
    PromptBuilder,
    PromptTemplate,
)
from rag.generation.strategies import RefineRAGStrategy, SimpleRAGStrategy
from rag.generation.utils import extract_texts, format_result_summary, truncate_text

__all__ = [
    "GenerationConfig",
    "OllamaClient",
    "ContextPreparer",
    "sanitize_context",
    "GenerationError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMTimeoutError",
    "GenerationResult",
    "PromptTemplate",
    "PromptBuilder",
    "STRICT_RAG_TEMPLATE",
    "COT_TEMPLATE",
    "FEW_SHOT_TEMPLATE",
    "REFINE_INITIAL_TEMPLATE",
    "REFINE_UPDATE_TEMPLATE",
    "BaseGenerationStrategy",
    "SimpleRAGStrategy",
    "RefineRAGStrategy",
    "extract_texts",
    "format_result_summary",
    "truncate_text",
]
