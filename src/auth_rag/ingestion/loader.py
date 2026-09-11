from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import requests

from auth_rag.ingestion.converter import BaseConverter, MarkdownConverter
from auth_rag.types import Document, Origin, RemoteDocument, Source

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Port for anything that produces Documents, regardless of the source."""

    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Yields the documents to feed into the ingestion pipeline, one at a time."""


class LocalLoader(BaseLoader):
    """Base for loaders that read Documents from data on the local machine."""

    def __init__(self, converter: BaseConverter | None = None) -> None:
        self.converter = converter or MarkdownConverter()


class FileLoader(LocalLoader):
    """Loads raw files from a directory, converting each to markdown by media type."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        converter: BaseConverter | None = None,
        save_output: bool = False,
    ) -> None:
        super().__init__(converter)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.save_output = save_output
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Iterator[Document]:
        for file_name in sorted(os.listdir(self.input_dir)):
            path = self.input_dir / file_name
            if not path.is_file():
                continue
            try:
                text = self.converter.convert_file(path)
            except Exception:
                logger.warning("Skipping file that could not be converted: %s", file_name, exc_info=True)
                continue
            if self.save_output:
                self._save_output(file_name, text)
            yield Document(
                text=text,
                # The file name is the stable identity of the document within its
                # origin dir (unique in a flat dir), mirroring external_id for remote
                # sources: it is the deletion key, stable across content edits (the
                # per-chunk id is content-hashed, so it changes when the text does).
                source=Source(id=file_name, name=file_name, origin=Origin.LOCAL, time=date.today()),
            )

    def _save_output(self, filename: str, text: str) -> None:
        output_path = self.output_dir / f"{filename}.md"
        output_path.write_text(text, encoding="utf-8")


class RemoteLoader(BaseLoader):
    """Base for loaders that pull Documents from an external service."""

    def __init__(self, converter: BaseConverter | None = None) -> None:
        self.converter = converter or MarkdownConverter()

    def load(self) -> Iterator[Document]:
        for record in self._fetch():
            try:
                item = RemoteDocument.model_validate(record)
                document = self._to_document(item)
            except Exception:
                logger.warning("Skipping remote record that could not be processed", exc_info=True)
                continue
            yield document

    @abstractmethod
    def _fetch(self) -> Iterable[dict]:
        """Fetches the raw records from the remote service (transport lives here)."""

    def _to_document(self, item: RemoteDocument) -> Document:
        """The mapping is the contract between an external service and the RAG types.

        ``origin`` is set here rather than read from the payload, so a service cannot claim
        it. The access attributes are copied as they arrived; the labeler checks them.
        """
        text = self.converter.convert_stream(item.data, item.media_type, name=item.title)
        return Document(
            text=text,
            source=Source(
                id=item.external_id,
                name=item.title,
                origin=Origin.REMOTE,
                time=item.time,
                access=item.access,
            ),
        )


class ApiLoader(RemoteLoader):
    """RemoteLoader that fetches documents from an HTTP JSON API.

    Owns the call: endpoint, auth headers, timeout. The response is expected to be
    a JSON list of records, or an object with an ``items`` list. Pagination, retries
    and auth refresh would be added here (``_fetch`` may yield across pages), without
    touching the contract or the mapping.
    """

    def __init__(
        self,
        base_url: str,
        path: str = "/documents",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        converter: BaseConverter | None = None,
    ) -> None:
        super().__init__(converter)
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.timeout = timeout
        self.headers = headers or {}

    def _fetch(self) -> Iterable[dict]:
        url = f"{self.base_url}{self.path}"
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload["items"] if isinstance(payload, dict) else payload
