from pathlib import Path

import pytest

from auth_rag.errors import ConversionError
from auth_rag.ingestion.converter import (
    BaseConverter,
    MarkdownConverter,
)


def test_markdown_converter_is_a_base_converter():
    assert isinstance(MarkdownConverter(), BaseConverter)


# --- convert: dispatch on media type (light paths only, no heavy parsers) ---

def test_text_payload_is_decoded_as_is():
    assert MarkdownConverter().convert_stream(b"ciao mondo", "text/plain") == "ciao mondo"


def test_markdown_payload_passes_through():
    assert MarkdownConverter().convert_stream(b"# Titolo", "text/markdown") == "# Titolo"


def test_csv_is_converted_to_a_markdown_table():
    out = MarkdownConverter().convert_stream(b"nome,eta\nMario,42", "text/csv", name="p.csv")
    assert "Mario" in out and "nome" in out


def test_unknown_media_type_fails_loud():
    with pytest.raises(ConversionError):
        MarkdownConverter().convert_stream(b"x", "application/octet-stream")


# --- convert_file: extension resolution (a source concern) ---

def test_convert_file_resolves_extension_case_insensitive(tmp_path):
    path = tmp_path / "note.TXT"
    path.write_text("ciao mondo")
    assert MarkdownConverter().convert_file(path) == "ciao mondo"


def test_convert_file_unknown_extension_fails_loud():
    with pytest.raises(ConversionError):
        MarkdownConverter().convert_file(Path("mystery.zzz"))
