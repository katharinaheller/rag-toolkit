from typing import Any, Dict, TypedDict


class Document(TypedDict):
    """A raw or cleaned document flowing through the pipeline."""
    id: str
    content: str
    metadata: Dict[str, Any]


class Chunk(TypedDict):
    """A fragment of a Document ready for embedding."""
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]
