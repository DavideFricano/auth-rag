"""What ``RelationIndex`` owns: turning a chunk into relations, and a traversal into a
score.

The graph is faked — ``_seeds`` and ``_expand`` return preset rows — so the family's two
shared decisions are exercised without an engine underneath. The engines are covered in
test_graph_index.py.
"""

from collections.abc import Iterable, Sequence
from datetime import date

import pytest

from auth_rag.indexing.relation.extractor import BaseExtractor, Triple
from auth_rag.indexing.relation.index import Relation, RelationIndex, normalize, terms
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Metadata, Origin, Source


class _StaticExtractor(BaseExtractor):
    """Returns the same triples for every chunk, and preset entities for every query."""

    def __init__(self, triples: list[Triple], named: list[str] | None = None) -> None:
        self.triples = triples
        self.named = named if named is not None else ["seed"]
        self.asked: list[str] = []

    def extract(self, chunk: Chunk) -> list[Triple]:
        return self.triples

    def entities(self, text: str) -> list[str]:
        self.asked.append(text)
        return self.named


class _FakeGraphIndex(RelationIndex):
    """Records what was added; expands to preset ``(chunk_id, hop, seed score)`` rows."""

    def __init__(self, store, extractor, rows=None, seeds=None, **kwargs) -> None:
        super().__init__(store, extractor, **kwargs)
        self.added: list[Relation] = []
        self.rows = rows or []
        self.seeds = seeds if seeds is not None else {"seed": 1.0}
        self.seen_terms: set[str] | None = None

    def _add(self, relations: Sequence[Relation]) -> None:
        self.added.extend(relations)

    def delete(self, source_id: str) -> None:  # not under test here
        raise NotImplementedError

    def _seeds(self, query_terms: set[str]) -> dict[str, float]:
        self.seen_terms = query_terms
        return self.seeds if query_terms else {}  # no terms, no way in — as a real tier

    def _expand(self, seeds: dict[str, float]) -> Iterable[tuple[str, int, float]]:
        return self.rows


def _chunk(id: str = "c0", source_id: str = "doc1") -> Chunk:
    source = Source(id=source_id, name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=f"text of {id}", metadata=Metadata(source=source, title="S"))


def _index(rows=None, seeds=None, triples=None, named=None, **kwargs) -> _FakeGraphIndex:
    return _FakeGraphIndex(
        VolatileStore(),
        _StaticExtractor(triples or [], named),
        rows=rows,
        seeds=seeds,
        **kwargs,
    )


# --- normalization, which is what makes two chunks meet on the same node ---------------


def test_normalize_folds_case_and_spacing():
    assert normalize("  Ipotiroidismo   DI  Grado 1 ") == "ipotiroidismo di grado 1"


def test_terms_drops_digits_and_very_short_words():
    assert terms("Tossicità da ICI e il T1") == {"tossicità", "ici"}


# --- insert ---------------------------------------------------------------------------


def test_insert_normalizes_names_and_anchors_the_relation_to_its_chunk():
    index = _index(triples=[Triple(subject="  Colite  ", predicate="treated by", object="STEROIDI")])
    index.insert([_chunk("c0", "doc1")])
    assert index.added == [
        Relation(
            subject="colite",
            predicate="treated by",
            object="steroidi",
            chunk_id="c0",
            source_id="doc1",
        )
    ]


def test_a_self_loop_is_dropped():
    """It bridges nothing: both ends are the same node, so no second chunk is reached."""
    index = _index(triples=[Triple(subject="Colite", predicate="is", object="colite")])
    index.insert([_chunk()])
    assert index.added == []


def test_a_relation_with_an_empty_end_is_dropped():
    index = _index(triples=[Triple(subject="   ", predicate="p", object="steroidi")])
    index.insert([_chunk()])
    assert index.added == []


def test_every_chunk_is_extracted():
    index = _index(triples=[Triple(subject="a", predicate="p", object="b")])
    index.insert([_chunk("c0"), _chunk("c1")])
    assert [relation.chunk_id for relation in index.added] == ["c0", "c1"]


# --- the score, which is the family's defining decision --------------------------------


def test_score_decays_with_distance_from_the_seed():
    index = _index(rows=[("c0", 0, 1.0), ("c1", 1, 1.0), ("c2", 2, 1.0)], decay=0.5)
    assert index._search("q", top_i=10) == [("c0", 1.0), ("c1", 0.5), ("c2", 0.25)]


def test_the_best_way_of_reaching_a_chunk_is_the_one_that_counts():
    """A chunk is usually reachable several ways; the expansion may yield it more than
    once, and a far path must not overwrite a near one."""
    index = _index(rows=[("c0", 2, 1.0), ("c0", 1, 1.0)], decay=0.5)
    assert index._search("q", top_i=10) == [("c0", 0.5)]


def test_a_weak_seed_nearby_can_lose_to_a_strong_seed_further_away():
    """Why the walk runs per seed instead of from one merged frontier."""
    index = _index(rows=[("c0", 1, 0.3), ("c0", 2, 0.9)], decay=0.5)
    assert index._search("q", top_i=10) == [("c0", 0.9 * 0.25)]


def test_no_seeds_means_no_results():
    index = _index(rows=[("c0", 0, 1.0)], seeds={})
    assert index._search("q", top_i=10) == []


# --- the entry point: the extractor reads the query, not the query itself --------------


def test_the_query_is_read_by_the_extractor():
    """The same component built the graph and now finds where to enter it, so both sides
    name concepts the same way instead of the entry inventing a second vocabulary."""
    index = _index(rows=[("c0", 0, 1.0)], named=["colite immunocorrelata"])
    index._search("qual è il trattamento della colite?", top_i=10)
    assert index.extractor.asked == ["qual è il trattamento della colite?"]


def test_the_matcher_sees_the_concept_words_not_the_users_words():
    """"trattamento" and "qual" never reach the matcher: only what the extractor named."""
    index = _index(rows=[("c0", 0, 1.0)], named=["colite immunocorrelata"])
    index._search("qual è il trattamento della colite?", top_i=10)
    assert index.seen_terms == {"colite", "immunocorrelata"}


def test_a_query_the_extractor_finds_nothing_in_opens_nothing():
    """The honest answer: this index has no entry point for that query. No fallback to
    splitting the query on spaces — that is the vocabulary we just stopped using."""
    index = _index(rows=[("c0", 0, 1.0)], named=[])
    assert index._search("qualunque cosa", top_i=10) == []
    assert index.seen_terms == set()


def test_ties_are_broken_by_id_so_the_answer_is_reproducible():
    index = _index(rows=[("c2", 0, 1.0), ("c0", 0, 1.0), ("c1", 0, 1.0)])
    assert [id_ for id_, _ in index._search("q", top_i=10)] == ["c0", "c1", "c2"]


# --- over-fetch ------------------------------------------------------------------------


def test_search_over_fetches_because_the_filter_thins_the_list_afterwards():
    """This index cannot filter — it does not know who is authorized — so it asks for
    more candidates than the caller's budget and lets ``retrieve`` do the cutting."""
    rows = [(f"c{i}", 0, 1.0 - i / 100) for i in range(30)]
    assert len(_index(rows=rows, over_fetch=3)._search("q", top_i=5)) == 15
    assert len(_index(rows=rows, over_fetch=1)._search("q", top_i=5)) == 5


# --- the knobs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hops": 0}, "hops must be at least 1"),
        ({"decay": 1.0}, "decay must lie in"),
        ({"decay": 0.0}, "decay must lie in"),
        ({"over_fetch": 0}, "over_fetch must be at least 1"),
    ],
)
def test_a_setting_that_would_make_the_expansion_meaningless_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _index(**kwargs)
