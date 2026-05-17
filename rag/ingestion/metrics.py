from dataclasses import dataclass, field
from typing import Callable

SkipSink = Callable[[], None]


@dataclass
class IngestionMetrics:
    """Aggregated counters for a single ingestion run."""

    docs_loaded: int = 0
    docs_dropped: int = 0
    chunks_produced: int = 0
    files_skipped: int = 0
    records_skipped: int = 0

    def skip_sink(self) -> SkipSink:
        """Return a bound callable that increments records_skipped by 1."""
        def _increment() -> None:
            self.records_skipped += 1
        return _increment

    def log(self, logger) -> None:
        """Emit a single structured log entry with all counters."""
        logger.info(
            "rag.ingestion.metrics",
            docs_loaded=self.docs_loaded,
            docs_dropped=self.docs_dropped,
            chunks_produced=self.chunks_produced,
            files_skipped=self.files_skipped,
            records_skipped=self.records_skipped,
        )

    def as_dict(self) -> dict:
        return {
            "docs_loaded": self.docs_loaded,
            "docs_dropped": self.docs_dropped,
            "chunks_produced": self.chunks_produced,
            "files_skipped": self.files_skipped,
            "records_skipped": self.records_skipped,
        }
