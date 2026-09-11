from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer
from markitdown import MarkItDown

from auth_rag.errors import ConversionError


class Parser(StrEnum):
    """Closed set of parsing strategies a media type can be routed to."""

    DOCLING = "docling"  # rich-layout documents (PDF, DOCX, PPTX, images)
    MARKITDOWN = "markitdown"  # structured / tabular / web (CSV, JSON, XLSX, HTML)
    TEXT = "text"  # already textual, decoded as-is


@dataclass(frozen=True)
class Format:
    """A supported format: its media type, the parser that handles it, and the
    file extensions it maps to (the first one is the representative extension)."""

    media_type: str
    parser: Parser
    extensions: tuple[str, ...]


# A file resolves its media type from the extension,
# a remote payload gets it from the gateway; both then dispatch on the media type.
# Adding a format is one row here — the lookup indexes below stay consistent.
FORMATS: tuple[Format, ...] = (
    Format("application/pdf", Parser.DOCLING, (".pdf",)),
    Format("application/vnd.openxmlformats-officedocument.wordprocessingml.document", Parser.DOCLING, (".docx",)),
    Format("application/vnd.openxmlformats-officedocument.presentationml.presentation", Parser.DOCLING, (".pptx",)),
    Format("image/png", Parser.DOCLING, (".png",)),
    Format("image/jpeg", Parser.DOCLING, (".jpg", ".jpeg")),
    Format("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", Parser.MARKITDOWN, (".xlsx",)),
    Format("text/csv", Parser.MARKITDOWN, (".csv",)),
    Format("application/json", Parser.MARKITDOWN, (".json",)),
    Format("text/html", Parser.MARKITDOWN, (".html", ".htm")),
    Format("text/markdown", Parser.TEXT, (".md",)),
    Format("text/plain", Parser.TEXT, (".txt",)),
)

PARSER_BY_MEDIA_TYPE = {f.media_type: f.parser for f in FORMATS}
EXTENSION_BY_MEDIA_TYPE = {f.media_type: f.extensions[0] for f in FORMATS}
MEDIA_TYPE_BY_EXTENSION = {ext: f.media_type for f in FORMATS for ext in f.extensions}


class BaseConverter(ABC):
    """Port for turning a raw payload into text, given its media type."""

    @abstractmethod
    def convert_stream(self, data: bytes, media_type: str, name: str = "document") -> str:
        """Converts an in-memory payload to text; ``name`` only aids format detection."""

    @abstractmethod
    def convert_file(self, path: Path) -> str:
        """Converts a file on disk to text (media type resolved from the path)."""


class MarkdownConverter(BaseConverter):
    """Turns a raw byte payload into markdown, dispatching on its media type."""

    def __init__(self, do_table: bool = True, do_ocr: bool = False) -> None:
        options = PdfPipelineOptions()
        options.do_table_structure = do_table
        options.do_ocr = do_ocr
        self.docling = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        self.markitdown = MarkItDown()


    def convert_stream(self, data: bytes, media_type: str, name: str = "document") -> str:
        ext = EXTENSION_BY_MEDIA_TYPE.get(media_type, "")
        match PARSER_BY_MEDIA_TYPE.get(media_type):
            case Parser.DOCLING:
                if not name.lower().endswith(ext):
                    name = f"{name}{ext}"
                result = self.docling.convert(DocumentStream(name=name, stream=BytesIO(data)))
                return result.document.export_to_markdown(included_content_layers={ContentLayer.BODY})
            case Parser.MARKITDOWN:
                return self.markitdown.convert_stream(BytesIO(data), file_extension=ext).text_content
            case Parser.TEXT:
                return data.decode("utf-8")
            case _:
                raise ConversionError(f"Unsupported media type: {media_type}")

    def convert_file(self, path: Path) -> str:
        media_type = MEDIA_TYPE_BY_EXTENSION.get(path.suffix.lower())
        match PARSER_BY_MEDIA_TYPE.get(media_type):
            case Parser.DOCLING:
                result = self.docling.convert(str(path))
                return result.document.export_to_markdown(included_content_layers={ContentLayer.BODY})
            case Parser.MARKITDOWN:
                return self.markitdown.convert(str(path)).text_content
            case Parser.TEXT:
                return path.read_text(encoding="utf-8")
            case _:
                raise ConversionError(f"Unsupported file extension: {path.name}")
