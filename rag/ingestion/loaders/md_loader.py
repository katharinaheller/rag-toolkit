from pathlib import Path
from typing import Iterable, Optional

from rag.ingestion.schema import Document
from rag.ingestion.loaders._text_reader import is_binary, read_text
from rag.ingestion.metrics import SkipSink
from rag.logging.logger import get_logger

logger = get_logger(__name__)


class MdLoader:

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".md"

    def load(self, path: Path, skip_sink: Optional[SkipSink] = None) -> Iterable[Document]:
        if is_binary(path):
            logger.info("loader.md.skip.binary", file=str(path))
            if skip_sink is not None:
                skip_sink()
            return

        text = read_text(path)
        if text is None:
            logger.info("loader.md.skip.unreadable", file=str(path))
            if skip_sink is not None:
                skip_sink()
            return

        yield Document(
            id="",
            content=text,
            metadata={"source": str(path), "type": "md", "format": "markdown"},
        )
