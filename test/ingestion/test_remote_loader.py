import base64
from collections.abc import Iterable
from datetime import date

from auth_rag.ingestion.loader import ApiLoader, RemoteLoader
from auth_rag.types import Document, Origin


def _item(**overrides):
    text = overrides.pop("text", "hello world")
    base = {
        "data": base64.b64encode(text.encode()).decode(),
        "media_type": "text/plain",
        "external_id": "doc-1",
        "title": "referto.pdf",
        "time": "2026-07-13",
    }
    base.update(overrides)
    return base


class _FakeRemoteLoader(RemoteLoader):
    """Exercises RemoteLoader's contract + mapping without any transport."""

    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self._records = records

    def _fetch(self) -> Iterable[dict]:
        return self._records


# --- RemoteLoader: contract + mapping (transport-independent) ---

def test_load_maps_remote_records_to_documents():
    loader = _FakeRemoteLoader([_item(), _item(external_id="doc-2", title="b.pdf")])

    docs = list(loader.load())

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    first = docs[0]
    assert first.text == "hello world"
    assert first.source.id == "doc-1"
    assert first.source.name == "referto.pdf"
    assert first.source.time == date(2026, 7, 13)
    assert first.source.origin == Origin.REMOTE


def test_load_decodes_base64_payload():
    docs = list(_FakeRemoteLoader([_item(text="referto del paziente")]).load())

    assert docs[0].text == "referto del paziente"


def test_load_is_tolerant_to_unknown_fields():
    docs = list(_FakeRemoteLoader([_item(extra_field="ignored", version=3)]).load())

    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_load_skips_malformed_records():
    incomplete = {"external_id": "x", "title": "t", "time": "2026-07-13"}  # no data/media_type

    docs = list(_FakeRemoteLoader([incomplete, _item()]).load())

    # one bad record is skipped, the good one still comes through
    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_load_skips_records_with_unsupported_media_type():
    bad = _item(external_id="doc-bad", media_type="application/octet-stream")

    docs = list(_FakeRemoteLoader([bad, _item(external_id="doc-2")]).load())

    assert {d.source.id for d in docs} == {"doc-2"}


def test_load_returns_empty_when_service_has_no_documents():
    assert list(_FakeRemoteLoader([]).load()) == []


# --- ApiLoader: the HTTP call ---

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_api_loader_calls_endpoint_and_maps(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return _FakeResponse({"items": [_item()]})

    monkeypatch.setattr("auth_rag.ingestion.loader.requests.get", fake_get)

    loader = ApiLoader("https://gw.example/", path="/documents", headers={"Authorization": "Bearer x"})
    docs = list(loader.load())

    assert captured["url"] == "https://gw.example/documents"  # trailing slash trimmed
    assert captured["headers"] == {"Authorization": "Bearer x"}
    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_api_loader_accepts_bare_list_payload(monkeypatch):
    monkeypatch.setattr(
        "auth_rag.ingestion.loader.requests.get",
        lambda *a, **k: _FakeResponse([_item(), _item(external_id="doc-2")]),
    )

    docs = list(ApiLoader("https://gw.example").load())

    assert {d.source.id for d in docs} == {"doc-1", "doc-2"}


# --- access attributes the service knows and the library cannot infer ---

def test_access_attributes_travel_into_the_source():
    """What the gateway knows about a record — patient, classification, care team — is
    what the library could never derive from the bytes, so it is carried across and lands
    on the document's Source, where every chunk will inherit it."""
    loader = _FakeRemoteLoader(
        [_item(access={"tenant": "asl-01", "patient_id": "P-4821", "care_team": ["cardiology"]})]
    )

    [doc] = list(loader.load())

    assert doc.source.access == {
        "tenant": "asl-01",
        "patient_id": "P-4821",
        "care_team": ["cardiology"],
    }


def test_a_record_without_access_yields_an_unlabeled_source():
    """No attributes is not an error here: whether that is acceptable is decided by the
    schema at labeling time, and by `required` at retrieval."""
    [doc] = list(_FakeRemoteLoader([_item()]).load())
    assert doc.source.access == {}


def test_the_service_cannot_claim_the_origin():
    """`origin` is set by the loader, not read from the payload: which channel a document
    came through is something the library knows, and a service must not be able to say
    otherwise — even by sending the field."""
    [doc] = list(_FakeRemoteLoader([_item(origin="local")]).load())
    assert doc.source.origin is Origin.REMOTE


def test_access_is_carried_verbatim_not_validated_here():
    """The loader transports, it does not check: an attribute nobody declared is refused
    by the labeler, which is the boundary that knows the vocabulary. Keeping the loader
    ignorant of it is what stops every loader from depending on the access schema."""
    [doc] = list(_FakeRemoteLoader([_item(access={"non_dichiarato": "x"})]).load())
    assert doc.source.access == {"non_dichiarato": "x"}
