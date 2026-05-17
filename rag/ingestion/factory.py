from rag.ingestion.composition import IngestionComponents
from rag.ingestion.loaders.registry import LoaderRegistry
from rag.ingestion.orchestrator import IngestionOrchestrator
from rag.ingestion.service import IngestionService


class IngestionFactory:

    @staticmethod
    def create_service(components: IngestionComponents) -> IngestionService:
        registry = LoaderRegistry(
            loaders=[(binding.extension, binding.loader) for binding in components.loader_bindings]
        )
        orchestrator = IngestionOrchestrator(
            loader_resolver=registry,
            cleaner=components.cleaner,
            chunker=components.chunker,
            doc_store=components.doc_store,
            chunk_store=components.chunk_store,
        )
        return IngestionService(orchestrator=orchestrator)
