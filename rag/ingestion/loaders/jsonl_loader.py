import json
from pathlib import Path
from typing import Iterable, Optional

from rag.ingestion.schema import Document
from rag.ingestion.metrics import SkipSink
from rag.logging.logger import get_logger

logger = get_logger(__name__)

_FALLBACK_KEYS = ("content", "body", "message")
_ENCODINGS = ("utf-8", "latin-1")


class JsonlLoader:
    """Streams JSONL records as Documents.

    Text field priority: "text" → "content" → "body" → "message" → full JSON.
    Accepts an optional skip_sink called once per skipped record.
    """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".jsonl"

    def _extract_text(self, obj: dict) -> Optional[str]:
        if isinstance(obj.get("text"), str):
            return obj["text"]
        for key in _FALLBACK_KEYS:
            if isinstance(obj.get(key), str):
                return obj[key]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return None

    def _iter_records(
        self, path: Path, encoding: str, skip_sink: Optional[SkipSink]
    ) -> Iterable[Document]:
        with path.open("r", encoding=encoding) as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.info("loader.jsonl.skip.malformed_line", file=str(path), line=line_no, error=str(exc))
                    if skip_sink is not None:
                        skip_sink()
                    continue

                text = self._extract_text(obj)
                if not text:
                    logger.info("loader.jsonl.skip.no_text_payload", file=str(path), line=line_no)
                    if skip_sink is not None:
                        skip_sink()
                    continue

                yield Document(
                    id="",
                    content=text,
                    metadata={"source": str(path), "type": "jsonl", "line": line_no},
                )

    def load(self, path: Path, skip_sink: Optional[SkipSink] = None) -> Iterable[Document]:
        for encoding in _ENCODINGS:
            try:
                yield from self._iter_records(path, encoding, skip_sink)
                return
            except UnicodeDecodeError:
                logger.info("loader.jsonl.retry_encoding", file=str(path), failed_encoding=encoding)
                continue
            except Exception as exc:
                logger.info("loader.jsonl.skip.unreadable", file=str(path), error=str(exc))
                if skip_sink is not None:
                    skip_sink()
                return

        logger.info("loader.jsonl.skip.all_encodings_failed", file=str(path))
        if skip_sink is not None:
            skip_sink()
