import hashlib
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field


def content_hash(text: str, length: int = 32) -> str:
    """Deterministic fingerprint of text content, used as a stable/idempotent id."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


AttributeValue = str | int | bool


class RemoteDocument(BaseModel):
    """The neutral raw payload the remote service guarantees.

    Carries bytes plus a media type, not text: turning the payload into text (PDF ->
    markdown, decode, ...) is the RAG side's job, so a single conversion path serves
    both filesystem and remote sources. Binary payloads travel base64-encoded in JSON
    and are decoded to real bytes on validation.

    ``access`` carries what the service knows and the library cannot infer from bytes —
    patient, classification, care team — keyed by the names the deployment's access schema
    declares. The labeler checks them; an undeclared one stops the ingestion.
    """

    model_config = ConfigDict(extra="ignore")

    data: Base64Bytes = Field(description="Raw document payload (base64-encoded in transit)")
    media_type: str = Field(description="IANA media type of the payload, e.g. application/pdf")
    external_id: str = Field(description="Stable id from the source system")
    title: str = Field(description="Human-readable document name")
    time: date = Field(description="When the service ingested the document")
    access: dict[str, AttributeValue | list[AttributeValue]] = Field(
        default_factory=dict,
        description="Access attributes the service knows, keyed by declared name",
    )


class Origin(StrEnum):
    """Where a document was acquired from."""

    LOCAL = "local"  # filesystem
    REMOTE = "remote"  # pulled from an external service


class Language(StrEnum):
    """Language for lexical tokenization. Values are the names the Snowball stemmer
    and NLTK stopwords expect, so a member can be passed to either directly."""

    ENGLISH = "english"
    ITALIAN = "italian"
    FRENCH = "french"
    GERMAN = "german"
    SPANISH = "spanish"


class Source(BaseModel):
    """Identifies the origin document of a chunk, and carries what it grants access to."""

    id: str = Field(description="Stable identifier of the document")
    name: str = Field(description="Name of the data source")
    origin: Origin = Field(description="Acquisition channel the document came from")
    time: date = Field(description="Date the source was loaded")
    access: dict[str, AttributeValue | list[AttributeValue]] = Field(
        default_factory=dict,
        description="Access attributes the policy filters on, keyed by declared name",
    )


class Document(BaseModel):
    """Document standard format as loaded."""

    text: str = Field(description="Full text of the original document")
    source: Source = Field(description="The document's source")


class Metadata(BaseModel):
    """Contextual metadata attached to a single chunk."""

    title: str = Field(description="Title of the document section the chunk was extracted from")
    source: Source = Field(description="Source of this specific chunk")
    page: int | None = Field(default=None, description="Optional page number")


class Chunk(BaseModel):
    """Atomic unit of text that flows through the retrieval pipeline."""

    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Text content of the chunk")
    metadata: Metadata = Field(description="Chunk metadata")


class ScoredChunk(BaseModel):
    """Chunk paired with a retrieval or ranking score."""

    chunk: Chunk = Field(description="The retrieved chunk")
    score: float = Field(description="Lexical/semantic relevance score")


class Message(BaseModel):
    """A single chat message: the neutral transport between augmentation and generation."""

    role: Literal["system", "user", "assistant"] = Field(description="Author role of the message")
    content: str = Field(description="Text content of the message")
