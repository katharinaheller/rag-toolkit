"""Tests for IngestionService.

The service is a thin adapter; its job is to delegate to the orchestrator
faithfully without altering behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from rag.ingestion.composition import IngestionRequest
from rag.ingestion.metrics import IngestionMetrics
from rag.ingestion.schema import Chunk
from rag.ingestion.service import IngestionService


class RecordingOrchestrator:
    """Captures kwargs that IngestionService forwards."""

    def __init__(self) -> None:
        self.calls = []

    def run(self, source: Path, persist: bool,
            metrics: IngestionMetrics | None = None) -> Iterator[Chunk]:
        self.calls.append({"source": source, "persist": persist, "metrics": metrics})
        yield {"id": "x", "document_id": "d", "text": "t", "metadata": {}}


class TestService:
    def test_ingest_forwards_args(self, tmp_path: Path) -> None:
        rec = RecordingOrchestrator()
        svc = IngestionService(orchestrator=rec)
        metrics = IngestionMetrics()
        list(svc.ingest(tmp_path, persist=True, metrics=metrics))
        assert rec.calls == [{"source": tmp_path, "persist": True, "metrics": metrics}]

    def test_ingest_request_unwraps_dataclass(self, tmp_path: Path) -> None:
        rec = RecordingOrchestrator()
        svc = IngestionService(orchestrator=rec)
        req = IngestionRequest(source=tmp_path, persist=False)
        list(svc.ingest_request(req))
        call = rec.calls[0]
        assert call["source"] == tmp_path
        assert call["persist"] is False
        assert call["metrics"] is None

    def test_ingest_yields_chunks(self, tmp_path: Path) -> None:
        rec = RecordingOrchestrator()
        svc = IngestionService(orchestrator=rec)
        chunks = list(svc.ingest(tmp_path, persist=False))
        assert chunks == [{"id": "x", "document_id": "d", "text": "t", "metadata": {}}]
