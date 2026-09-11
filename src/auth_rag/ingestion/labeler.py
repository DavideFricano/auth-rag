from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from auth_rag.authorization.schema import AccessSchema
from auth_rag.errors import ConformanceError, DeclarationError
from auth_rag.types import Document


class BaseLabeler(ABC):
    """Writes the access attributes of a document, at the ingestion boundary.

    **Labelling is not deciding.** It marks the data for *what it is*, never for *who may
    see it*, so changing a policy never forces a re-ingestion.

    Runs on the ``Document``, before chunking: the attributes are written once on its
    ``Source`` and every chunk inherits them by carrying it. Subclasses only say where the
    values come from (``_attributes``); validating and attaching them is shared.
    """

    def __init__(self, schema: AccessSchema) -> None:
        self.schema = schema

    @abstractmethod
    def _attributes(self, document: Document) -> Mapping[str, object]:
        """The raw attributes for this document, before validation."""

    def label(self, document: Document) -> Document:
        """A copy of the document whose ``source.access`` holds the validated attributes.

        Attributes that are invalid or missing raise rather than being skipped: unlike a
        corrupt file, they mean the producer and the declaration disagree. The failure
        carries the document id, which the schema cannot know.
        """
        try:
            access = self.schema.validate_access(self._attributes(document))
        except ConformanceError as error:
            # the schema knows the vocabulary, only this side knows which document
            raise ConformanceError(f"document {document.source.id!r}: {error}") from error
        return document.model_copy(
            update={"source": document.source.model_copy(update={"access": access})}
        )


class PropagatingLabeler(BaseLabeler):
    """Keeps the attributes the document already arrived with, and validates them.

    The main path: whoever produced the corpus knows the attributes and delivers them, so
    this adds no values — it is the boundary where what came from outside is checked.
    """

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return document.source.access


class ManifestLabeler(BaseLabeler):
    """Reads the attributes from a JSON manifest keyed by ``source.id``.

    For a corpus with no producer upstream, curated by hand. ``default`` applies to every
    document and ``sources`` overrides it per document, so nothing has to be enumerated in
    advance.

    ``{"default": {"tenant": "acme"}, "sources": {"report.pdf": {"classification": "…"}}}``
    """

    def __init__(self, schema: AccessSchema, path: Path | str) -> None:
        super().__init__(schema)
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise DeclarationError(f"the access manifest must be a JSON object: {path}")
        unknown = set(manifest) - {"default", "sources"}
        if unknown:
            raise DeclarationError(f"unknown key in the access manifest: {sorted(unknown)}")
        self.default: dict = manifest.get("default", {})
        self.sources: dict = manifest.get("sources", {})

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return {**self.default, **self.sources.get(document.source.id, {})}


class StaticLabeler(BaseLabeler):
    """Puts the same attributes on everything this pipeline ingests.

    An ``IngestionPipeline`` has a single loader, so a corpus that shares its attributes
    needs nothing more. Several sources means several pipelines, each with its own labeler.
    """

    def __init__(self, schema: AccessSchema, attributes: Mapping[str, object]) -> None:
        super().__init__(schema)
        self.attributes = dict(attributes)

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return self.attributes
