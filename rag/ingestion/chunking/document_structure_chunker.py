from typing import Iterable, List

import marko
import marko.inline as minline

from rag.ingestion.schema import Chunk, Document
from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.core import build_chunk
from rag.ingestion.chunking.strategies import ChunkingStrategy


def _inline_text(node) -> str:
    """Recursively extract inline text from a marko inline node."""
    if isinstance(node.children, str):
        return node.children
    parts = []
    for child in node.children:
        if isinstance(child, minline.RawText):
            parts.append(child.children)
        elif isinstance(child, minline.CodeSpan):
            parts.append(f"`{child.children}`")
        elif isinstance(child, minline.LineBreak):
            parts.append("\n")
        elif isinstance(child, minline.SoftBreak):
            parts.append(" ")
        elif hasattr(child, "children"):
            parts.append(_inline_text(child))
    return "".join(parts)


def _node_to_text(node) -> str:
    """Convert a marko block node to clean, embedding-friendly text."""
    name = type(node).__name__

    if name == "Heading":
        return "#" * node.level + " " + _inline_text(node) + "\n"
    if name == "Paragraph":
        return _inline_text(node) + "\n"
    if name in ("FencedCode", "CodeBlock"):
        lang = getattr(node, "lang", "") or ""
        code = node.children[0].children if node.children else ""
        fence = f"```{lang}\n" if lang else "```\n"
        return fence + code + "```\n"
    if name == "List":
        lines = []
        for i, item in enumerate(node.children):
            prefix = f"{i + 1}. " if node.ordered else "- "
            inner = "".join(_node_to_text(c) for c in item.children).strip()
            lines.append(prefix + inner)
        return "\n".join(lines) + "\n"
    if name == "BlockQuote":
        inner = "".join(_node_to_text(c) for c in node.children)
        return "\n".join("> " + line for line in inner.splitlines()) + "\n"
    if name == "ThematicBreak":
        return "---\n"
    if hasattr(node, "children"):
        if isinstance(node.children, str):
            return node.children
        return "".join(_node_to_text(c) for c in node.children)
    return ""


class DocumentStructureChunker(Chunker):
    """Structure-aware chunker for Markdown.

    Falls back to fallback_chunker when parsing fails or the document is empty.
    Fallback chunks carry metadata["fallback"] = True.
    """

    def __init__(
        self,
        max_chunk_size: int,
        fallback_chunker: Chunker,
        strategy: ChunkingStrategy,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.fallback_chunker = fallback_chunker
        self.strategy = strategy

    def _parse_blocks(self, text: str) -> List[str]:
        parsed = marko.parse(text)
        blocks = []
        for child in parsed.children:
            rendered = _node_to_text(child).strip()
            if rendered:
                blocks.append(rendered)
        return blocks

    def _fallback(self, doc: Document) -> Iterable[Chunk]:
        for chunk in self.fallback_chunker.chunk(doc):
            yield {
                **chunk,
                "metadata": {**chunk["metadata"], "strategy": self.strategy.name, "fallback": True},
            }

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        try:
            blocks = self._parse_blocks(doc["content"])
        except Exception:
            yield from self._fallback(doc)
            return

        if not blocks:
            yield from self._fallback(doc)
            return

        current: list[str] = []
        size = 0
        idx = 0

        for block in blocks:
            block_len = len(block)
            if current and size + block_len > self.max_chunk_size:
                yield build_chunk(
                    document_id=doc["id"],
                    index=idx,
                    text="\n\n".join(current),
                    base_metadata=doc["metadata"],
                    extra_metadata={},
                    strategy=self.strategy,
                )
                idx += 1
                current = []
                size = 0
            current.append(block)
            size += block_len

        if current:
            yield build_chunk(
                document_id=doc["id"],
                index=idx,
                text="\n\n".join(current),
                base_metadata=doc["metadata"],
                extra_metadata={},
                strategy=self.strategy,
            )
