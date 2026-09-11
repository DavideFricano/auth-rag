from datetime import date

import pytest
from pydantic import ValidationError

from auth_rag.types import Chunk, Metadata, Origin, ScoredChunk, Source


def test_source_valid():
    s = Source(id="doc1", name="file.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    assert s.name == "file.pdf"


def test_source_missing_field_raises():
    with pytest.raises(ValidationError):
        Source(id="doc1", name="file.pdf")


def test_metadata_page_defaults_to_none():
    s = Source(id="doc1", name="file.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    m = Metadata(source=s, title="Intro")
    assert m.page is None


def test_chunk_valid():
    s = Source(id="doc1", name="file.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    m = Metadata(source=s, title="Intro")
    c = Chunk(id="c0", text="Testo.", metadata=m)
    assert c.id == "c0"


def test_scored_chunk_missing_score_raises():
    s = Source(id="doc1", name="file.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    m = Metadata(source=s, title="Intro")
    c = Chunk(id="c0", text="Testo.", metadata=m)
    with pytest.raises(ValidationError):
        ScoredChunk(chunk=c)
