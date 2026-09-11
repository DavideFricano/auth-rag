"""The authorization check in ``BaseIndex.retrieve``.

``evaluate`` is covered in isolation (test/authorization/test_filter.py); what is tested
here is the wiring around it — that the predicate is applied to the chunks resolved from
the shared store, on the base class, for every index. The backend is faked so the check
is exercised on its own terms: a fake ``_search`` returns preset ``(id, score)`` rows and
knows nothing about the filter, which is exactly the case the base class must cover — an
index that ignores or mistranslates a pushdown must still not leak.
"""

from datetime import date

import pytest

from auth_rag.authorization.filter import Allow, And, Match, Not
from auth_rag.authorization.schema import AccessSchema, Attribute, AttributeType
from auth_rag.errors import ConformanceError, EnforcementError
from auth_rag.indexing.index import BaseIndex
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Metadata, Origin, Source


class _FakeIndex(BaseIndex):
    """Index whose backend returns preset hits and never sees the filter."""

    def __init__(
        self,
        store: VolatileStore,
        hits: list[tuple[str, float]],
        schema: AccessSchema | None = None,
    ) -> None:
        super().__init__(store, schema)
        self.hits = hits
        self.searched_top_i: int | None = None

    def insert(self, chunks: list[Chunk]) -> None:  # not under test here
        raise NotImplementedError

    def delete(self, source_id: str) -> None:  # not under test here
        raise NotImplementedError

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        self.searched_top_i = top_i
        return self.hits[:top_i]


def _chunk(id: str, access: dict | None = None) -> Chunk:
    source = Source(
        id="doc1", name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1), access=access or {}
    )
    return Chunk(id=id, text=f"text of {id}", metadata=Metadata(source=source, title="S"))


_ACME = Match(attribute="tenant", values={"acme"})

# A deployment that declared ABAC: `tenant` must be on every chunk, `classification` may be.
_SCHEMA = AccessSchema(
    [
        Attribute(name="tenant", type=AttributeType.KEYWORD, required=True),
        Attribute(name="classification", type=AttributeType.KEYWORD),
    ]
)


def _wired(
    chunks: list[Chunk], hits: list[tuple[str, float]], schema: AccessSchema | None = _SCHEMA
) -> _FakeIndex:
    """Declared schema by default, since filtering now requires one; pass ``schema=None``
    for the deployment that does no access control at all."""
    store = VolatileStore()
    store.add(chunks)
    return _FakeIndex(store, hits, schema)


def test_unauthorized_chunks_are_dropped():
    index = _wired(
        [_chunk("c0", {"tenant": "acme"}), _chunk("c1", {"tenant": "globex"})],
        [("c0", 0.9), ("c1", 0.8)],
    )
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10, filter=_ACME)] == ["c0"]


def test_the_check_holds_when_the_backend_ignores_the_filter():
    """The fake backend returns the unauthorized hit first and knows nothing about the
    predicate — as a backend without pushdown, or with a mistranslated one, would. The
    base class must still be what decides, so a bad translation costs recall, not secrecy.
    """
    index = _wired(
        [_chunk("c0", {"tenant": "globex"}), _chunk("c1", {"tenant": "acme"})],
        [("c0", 0.99), ("c1", 0.10)],
    )
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10, filter=_ACME)] == ["c1"]


def test_attributes_come_from_the_store_not_from_the_index():
    """The access attributes ride on the chunk's ``Source``, part of the record only the
    store holds — the index payload carries ids. Filtering therefore has to happen after
    the resolution, and a chunk the store knows nothing about cannot be authorized.
    """
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9), ("ghost", 0.8)])
    results = index.retrieve("q", top_i=10, filter=_ACME)
    assert [sc.chunk.id for sc in results] == ["c0"]
    assert results[0].chunk.metadata.source.access == {"tenant": "acme"}


def test_unlabeled_chunk_is_denied():
    """Default-deny at the enforcement point, not only inside ``evaluate``: a chunk the
    labeler never tagged carries an empty ``access`` and must not pass any filter."""
    index = _wired([_chunk("c0"), _chunk("c1", {"tenant": "acme"})], [("c0", 0.9), ("c1", 0.8)])
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10, filter=_ACME)] == ["c1"]


def test_a_composite_predicate_is_honoured_whole():
    """The whole tree decides, not just its first clause."""
    not_confidential = Not(clause=Match(attribute="classification", values={"confidential"}))
    predicate = And(clauses=[_ACME, not_confidential])
    index = _wired(
        [
            _chunk("c0", {"tenant": "acme", "classification": "public"}),
            _chunk("c1", {"tenant": "acme", "classification": "confidential"}),
            _chunk("c2", {"tenant": "globex", "classification": "public"}),
        ],
        [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)],
    )
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10, filter=predicate)] == ["c0"]


def test_survivors_keep_their_backend_score_and_relative_order():
    """The filter removes rows, it must not rescore or reorder them: the fusion ranker
    reads the positions and the score spread of this list."""
    index = _wired(
        [
            _chunk("c0", {"tenant": "acme"}),
            _chunk("c1", {"tenant": "globex"}),
            _chunk("c2", {"tenant": "acme"}),
        ],
        [("c0", 0.9), ("c1", 0.8), ("c2", 0.7)],
    )
    results = index.retrieve("q", top_i=10, filter=_ACME)
    assert [(sc.chunk.id, sc.score) for sc in results] == [("c0", 0.9), ("c2", 0.7)]


def test_denying_everything_returns_an_empty_list():
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9)])
    deny = Match(attribute="tenant", values=set())
    assert index.retrieve("q", top_i=10, filter=deny) == []


def test_labelled_chunks_read_by_an_index_that_enforces_nothing_are_refused():
    """Someone labelled the data and this index was wired without the schema, so the read
    would hand back every tenant at once. The other direction is refused too, one test
    below: schema and filter go together."""
    index = _wired(
        [_chunk("c0", {"tenant": "acme"}), _chunk("c1", {"tenant": "globex"})],
        [("c0", 0.9), ("c1", 0.8)],
        schema=None,
    )
    with pytest.raises(EnforcementError, match="enforces nothing"):
        index.retrieve("q", top_i=10)


def test_filtering_without_a_declared_schema_is_refused():
    """The two go together. Without a vocabulary the predicate cannot be validated and
    there is no notion of a required attribute, so a bare negation would admit a chunk
    nobody labelled — a filter that quietly guarantees less than it looks like it does."""
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9)], schema=None)
    with pytest.raises(EnforcementError, match="filtering needs the AccessSchema"):
        index.retrieve("q", top_i=10, filter=_ACME)


def test_without_a_schema_the_filter_stays_optional():
    """A deployment that declares no schema does no ABAC, and nothing changes for it:
    omitting the filter is not an error, it returns everything."""
    index = _wired([_chunk("c0"), _chunk("c1")], [("c0", 0.9), ("c1", 0.8)], schema=None)
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10)] == ["c0", "c1"]


def test_with_a_schema_omitting_the_filter_is_refused():
    """The safe default cannot be inferred here, and forgetting an argument must not be
    spelled the same way as deciding there is no restriction."""
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9)], _SCHEMA)
    with pytest.raises(EnforcementError, match="requires a filter"):
        index.retrieve("q", top_i=10)


def test_the_refusal_happens_before_the_backend_is_queried():
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9)], _SCHEMA)
    with pytest.raises(EnforcementError):
        index.retrieve("q", top_i=10)
    assert index.searched_top_i is None


def test_allow_is_how_a_call_states_it_has_no_restriction():
    index = _wired(
        [_chunk("c0", {"tenant": "acme"}), _chunk("c1", {"tenant": "globex"})],
        [("c0", 0.9), ("c1", 0.8)],
        _SCHEMA,
    )
    assert [sc.chunk.id for sc in index.retrieve("q", top_i=10, filter=Allow())] == ["c0", "c1"]


def test_a_chunk_missing_a_required_attribute_is_denied_under_any_predicate():
    """The hole this check closes: under a bare Not, a missing attribute makes the inner
    Match false and the negation true, so an unlabeled chunk would pass. Allow() is
    covered too — it waives the policy, not the schema's integrity.
    """
    index = _wired(
        [_chunk("c0", {"tenant": "acme"}), _chunk("unlabeled")],
        [("c0", 0.9), ("unlabeled", 0.8)],
        _SCHEMA,
    )
    not_confidential = Not(clause=Match(attribute="classification", values={"confidential"}))
    assert [sc.chunk.id for sc in index.retrieve("q", 10, filter=not_confidential)] == ["c0"]
    assert [sc.chunk.id for sc in index.retrieve("q", 10, filter=Allow())] == ["c0"]


def test_a_filter_naming_an_undeclared_attribute_is_refused():
    """It would match nothing and return an empty list, which reads like 'you have no
    access' instead of 'this filter is wrong'. The vocabulary makes it loud."""
    index = _wired([_chunk("c0", {"tenant": "acme"})], [("c0", 0.9)], _SCHEMA)
    with pytest.raises(ConformanceError, match="undeclared access attribute"):
        index.retrieve("q", top_i=10, filter=Match(attribute="departement", values={"x"}))


def test_the_filter_can_return_fewer_than_top_i():
    """``top_i`` is spent on the backend before the filter runs, so an authorized result
    sitting below the cut is lost. Documents today's behaviour: the over-fetch the roadmap
    calls for is not implemented, so this is where recall is paid for."""
    index = _wired(
        [_chunk("c0", {"tenant": "globex"}), _chunk("c1", {"tenant": "acme"})],
        [("c0", 0.9), ("c1", 0.8)],
    )
    assert index.retrieve("q", top_i=1, filter=_ACME) == []
    assert index.searched_top_i == 1
