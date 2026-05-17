from typing import Iterator

from rag.ingestion.composition import IngestionComponents, IngestionRequest
from rag.ingestion.factory import IngestionFactory
from rag.ingestion.metrics import IngestionMetrics
from rag.ingestion.schema import Chunk
from rag.ingestion.service import IngestionService


def create_ingestion_service(components: IngestionComponents) -> IngestionService:
    return IngestionFactory.create_service(components)


def run_ingestion(
    service: IngestionService,
    request: IngestionRequest,
    metrics: IngestionMetrics | None = None,
) -> Iterator[Chunk]:
    return service.ingest_request(request, metrics=metrics)
