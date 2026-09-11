"""The labeler: where the access attributes are written and checked at ingestion.

What matters here is the boundary, not the values: whatever a labeler decides to put on a
document goes through the declared vocabulary before it can reach the store, and it lands
on the document's ``Source`` so that every chunk inherits it without anyone copying.
"""

from datetime import date

import pytest

from auth_rag.authorization.schema import AccessSchema, Attribute, AttributeType
from auth_rag.errors import ConformanceError, DeclarationError
from auth_rag.ingestion.chunker import FixedSizeChunker
from auth_rag.ingestion.labeler import ManifestLabeler, PropagatingLabeler, StaticLabeler
from auth_rag.types import Document, Origin, Source

_SCHEMA = AccessSchema(
    [
        Attribute(name="tenant", type=AttributeType.KEYWORD, required=True),
        Attribute(name="care_team", type=AttributeType.KEYWORD, multi=True),
        Attribute(name="retention_years", type=AttributeType.INTEGER),
    ]
)


def _document(id: str = "report.pdf", access: dict | None = None) -> Document:
    source = Source(
        id=id, name=id, origin=Origin.LOCAL, time=date(2024, 1, 1), access=access or {}
    )
    return Document(text="testo del documento", source=source)


def _manifest(tmp_path, text: str) -> str:
    path = tmp_path / "access.json"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_propagating_keeps_what_arrived_with_the_document():
    """The main path: the producer upstream knows the attributes, the library checks them
    against the declaration and adds nothing of its own."""
    document = _document(access={"tenant": "acme", "care_team": ["icu"]})
    labeled = PropagatingLabeler(_SCHEMA).label(document)
    assert labeled.source.access == {"tenant": "acme", "care_team": ["icu"]}


def test_an_undeclared_attribute_stops_the_ingestion():
    """Not skipped like a corrupt file: an attribute nobody declared means the producer
    and the declaration disagree, which concerns the whole corpus and not this document.
    Skipping would leave a silent hole in it."""
    labeler = PropagatingLabeler(_SCHEMA)
    with pytest.raises(ConformanceError, match="undeclared access attribute"):
        labeler.label(_document(access={"departement": "cardio"}))


def test_a_wrongly_typed_value_stops_the_ingestion():
    labeler = PropagatingLabeler(_SCHEMA)
    with pytest.raises(ConformanceError, match="retention_years"):
        labeler.label(_document(access={"tenant": "acme", "retention_years": "dieci"}))


def test_the_document_is_not_mutated():
    """``label`` returns a copy: a caller holding the loaded document keeps what it read."""
    document = _document(access={"tenant": "acme"})
    StaticLabeler(_SCHEMA, {"tenant": "globex"}).label(document)
    assert document.source.access == {"tenant": "acme"}


def test_static_puts_the_same_attributes_on_every_document():
    labeler = StaticLabeler(_SCHEMA, {"tenant": "acme"})
    assert labeler.label(_document("a.pdf")).source.access == {"tenant": "acme"}
    assert labeler.label(_document("b.pdf")).source.access == {"tenant": "acme"}


def test_static_validates_too():
    with pytest.raises(ConformanceError, match="undeclared access attribute"):
        StaticLabeler(_SCHEMA, {"nope": "x"}).label(_document())


def test_manifest_overrides_the_default_per_source(tmp_path):
    path = _manifest(
        tmp_path,
        """
        {
          "default": { "tenant": "acme" },
          "sources": { "referto.pdf": { "tenant": "globex", "care_team": ["icu"] } }
        }
        """,
    )
    labeler = ManifestLabeler(_SCHEMA, path)
    assert labeler.label(_document("altro.pdf")).source.access == {"tenant": "acme"}
    assert labeler.label(_document("referto.pdf")).source.access == {
        "tenant": "globex",
        "care_team": ["icu"],
    }


def test_manifest_does_not_need_every_document_enumerated(tmp_path):
    """A document nobody listed inherits the default, so a growing corpus does not have to
    be enumerated in advance to stay ingestible."""
    path = _manifest(
        tmp_path,
        '{ "default": { "tenant": "acme" }, "sources": { "referto.pdf": { "tenant": "globex" } } }',
    )
    labeled = ManifestLabeler(_SCHEMA, path).label(_document("nuovo.pdf"))
    assert labeled.source.access == {"tenant": "acme"}


def test_a_document_the_default_does_not_cover_stops_the_ingestion(tmp_path):
    """With no default carrying the required attributes, an unlisted document could never
    be retrieved. Indexing it anyway would defer the symptom to query time, where an empty
    result is indistinguishable from a legitimate deny — so it is refused here, naming the
    document, which is the one thing the schema cannot know."""
    path = _manifest(tmp_path, '{ "sources": { "referto.pdf": { "tenant": "acme" } } }')
    with pytest.raises(ConformanceError, match="nuovo.pdf"):
        ManifestLabeler(_SCHEMA, path).label(_document("nuovo.pdf"))


def test_an_unknown_key_in_the_manifest_is_refused(tmp_path):
    path = _manifest(tmp_path, '{ "defaults": { "tenant": "acme" } }')
    with pytest.raises(DeclarationError, match="unknown key"):
        ManifestLabeler(_SCHEMA, path)


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path):
    with pytest.raises(DeclarationError, match="JSON object"):
        ManifestLabeler(_SCHEMA, _manifest(tmp_path, '[{ "tenant": "acme" }]'))


def test_every_chunk_of_the_document_inherits_the_attributes():
    """The reason labeling happens before chunking and on the Source: the chunker copies
    ``doc.source`` into each chunk's metadata, so one write reaches them all."""
    labeled = StaticLabeler(_SCHEMA, {"tenant": "acme"}).label(
        Document(text="x" * 900, source=_document().source)
    )
    chunks = FixedSizeChunker(chunk_size=100, overlap=0).chunk(labeled)
    assert len(chunks) > 1
    assert all(chunk.metadata.source.access == {"tenant": "acme"} for chunk in chunks)


def test_a_multi_valued_attribute_must_arrive_as_a_collection():
    with pytest.raises(ConformanceError, match="care_team"):
        PropagatingLabeler(_SCHEMA).label(
            _document(access={"tenant": "acme", "care_team": "icu"})
        )
