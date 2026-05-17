from pathlib import Path
from typing import Iterator

from rag.ingestion.composition import IngestionRequest
from rag.ingestion.metrics import IngestionMetrics
from rag.ingestion.orchestrator import IngestionOrchestrator
from rag.ingestion.schema import Chunk


class IngestionService:
    """Thin adapter bridging the public API to the orchestrator."""

    def __init__(self, orchestrator: IngestionOrchestrator) -> None:
        self.orchestrator = orchestrator

    def ingest(
        self,
        source: Path,
        persist: bool,
        metrics: IngestionMetrics | None = None,
    ) -> Iterator[Chunk]:
        return self.orchestrator.run(source=source, persist=persist, metrics=metrics)

    def ingest_request(
        self,
        request: IngestionRequest,
        metrics: IngestionMetrics | None = None,
    ) -> Iterator[Chunk]:
        return self.ingest(source=request.source, persist=request.persist, metrics=metrics)
