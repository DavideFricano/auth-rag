"""The engines, and the claim that they are interchangeable.

Most of this file runs **against both tiers**: the docs say tiers differ only in the
technology underneath and never in behaviour, and for the graph that is a strong claim —
an in-process breadth-first walk and a Cypher traversal share no code at all. So the
behaviour is asserted once and parametrized over the two.

The networkx tier is the real thing in-process, the way ``:memory:`` Qdrant is for the
similarity family. The Neo4j tier needs a server and is skipped unless one is pointed at::

    docker run -d -p 7688:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5-community
    AUTH_RAG_TEST_NEO4J=bolt://localhost:7688 \\
    AUTH_RAG_TEST_NEO4J_PASSWORD=testpassword uv run pytest
"""

import os
from datetime import date

import pytest

from auth_rag.authorization.filter import Allow, Match
from auth_rag.authorization.schema import AccessSchema, Attribute, AttributeType
from auth_rag.indexing.relation.extractor import BaseExtractor, Triple
from auth_rag.indexing.relation.graph_index import RemoteGraphIndex, VolatileGraphIndex
from auth_rag.storing.store import VolatileStore
from auth_rag.types import Chunk, Metadata, Origin, Source


class _ScriptedExtractor(BaseExtractor):
    """Gives each chunk the triples written for it, by chunk id.

    ``entities`` stands in for the query side: it recognizes the concept names its own
    script knows, which is what a real extractor does when it reads a question — name
    concepts rather than repeat the user's words.
    """

    def __init__(self, script: dict[str, list[tuple[str, str, str]]]) -> None:
        self.script = script
        self.known = {name for triples in script.values() for s, _, o in triples for name in (s, o)}

    def extract(self, chunk: Chunk) -> list[Triple]:
        return [
            Triple(subject=s, predicate=p, object=o) for s, p, o in self.script.get(chunk.id, [])
        ]

    def entities(self, text: str) -> list[str]:
        return sorted(name for name in self.known if name.lower() in text.lower())


_SCHEMA = AccessSchema([Attribute(name="tenant", type=AttributeType.KEYWORD, required=True)])


def _chunk(id: str, source_id: str = "doc1", access: dict | None = None) -> Chunk:
    source = Source(
        id=source_id,
        name=f"{source_id}.pdf",
        origin=Origin.LOCAL,
        time=date(2024, 1, 1),
        access=access or {},
    )
    return Chunk(id=id, text=f"text of {id}", metadata=Metadata(source=source, title="S"))


@pytest.fixture(params=["volatile", "remote"])
def build(request):
    """A factory for one tier: same arguments, same expected behaviour."""
    if request.param == "volatile":

        def volatile(chunks, script, schema=None, **kwargs):
            store = VolatileStore()
            store.add(chunks)
            index = VolatileGraphIndex(store, _ScriptedExtractor(script), schema=schema, **kwargs)
            index.insert(chunks)
            return index

        yield volatile
        return

    url = os.getenv("AUTH_RAG_TEST_NEO4J")
    if not url:
        pytest.skip("needs a Neo4j (set AUTH_RAG_TEST_NEO4J=<bolt url>)")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        url,
        auth=(
            os.getenv("AUTH_RAG_TEST_NEO4J_USER", "neo4j"),
            os.getenv("AUTH_RAG_TEST_NEO4J_PASSWORD", "neo4j"),
        ),
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # each test starts on an empty graph

    def remote(chunks, script, schema=None, **kwargs):
        store = VolatileStore()
        store.add(chunks)
        index = RemoteGraphIndex(
            store, _ScriptedExtractor(script), driver=driver, schema=schema, **kwargs
        )
        index.insert(chunks)
        return index

    yield remote
    driver.close()


# --- the thesis, on both tiers ---------------------------------------------------------


def test_a_shared_concept_bridges_two_chunks(build):
    """``c1`` never mentions the query's words. It is reached because both chunks named
    "colite", which is the whole mechanism: nodes merge by name."""
    index = build(
        [_chunk("c0"), _chunk("c1")],
        {
            "c0": [("colite", "treated by", "steroidi")],
            "c1": [("colite", "graded by", "ctcae")],
        },
    )
    found = dict(index._search("steroidi", top_i=10))
    assert set(found) == {"c0", "c1"}
    assert found["c0"] > found["c1"]  # the seed's own chunk outranks the one a hop away


def test_hops_bound_how_far_the_expansion_reaches(build):
    script = {
        "c0": [("alpha", "p", "beta")],
        "c1": [("beta", "p", "gamma")],
        "c2": [("gamma", "p", "delta")],
    }
    chunks = [_chunk("c0"), _chunk("c1"), _chunk("c2")]
    assert set(dict(build(chunks, script, hops=1)._search("alpha", top_i=10))) == {"c0", "c1"}
    assert set(dict(build(chunks, script, hops=3)._search("alpha", top_i=10))) == {
        "c0",
        "c1",
        "c2",
    }


def test_a_query_the_extractor_finds_no_concept_in_opens_nothing(build):
    """The honest answer for this index: it has no entry point for that query. The
    similarity indexes are still answering, so the pipeline is not left empty."""
    index = build([_chunk("c0")], {"c0": [("colite", "p", "steroidi")]})
    assert index._search("pneumotorace", top_i=10) == []


def test_the_query_enters_through_the_extractor_not_through_its_own_words(build):
    """The words the *user* chose never reach the matcher — only the concepts the
    extractor named. Here "trattamento" is not a node, and the sentence around it is
    noise, yet the query still opens the graph on "colite"."""
    index = build(
        [_chunk("c0"), _chunk("c1")],
        {"c0": [("colite", "trattata con", "steroidi")], "c1": [("colite", "p", "ctcae")]},
    )
    assert set(dict(index._search("qual è il trattamento della colite?", top_i=10))) == {
        "c0",
        "c1",
    }


def test_case_and_spacing_do_not_decide_whether_a_bridge_exists(build):
    """Two chunks spelling the same concept differently must still meet on one node."""
    index = build(
        [_chunk("c0"), _chunk("c1")],
        {"c0": [("Colite", "p", "steroidi")], "c1": [("  COLITE  ", "p", "ctcae")]},
    )
    assert set(dict(index._search("steroidi", top_i=10))) == {"c0", "c1"}


def test_seeds_prefer_the_concept_the_query_names_outright(build):
    """A one-word concept the query hits entirely beats a long one it only clipped."""
    index = build(
        [_chunk("c0")],
        {"c0": [("colite", "p", "colite immunocorrelata di grado tre")]},
    )
    assert index._seeds({"colite"}) == {
        "colite": 1.0,
        # four terms, not five: "di" is below the length cut-off and never counted
        "colite immunocorrelata di grado tre": pytest.approx(0.25),
    }


def test_max_seeds_caps_how_many_entry_points_a_query_opens(build):
    script = {"c0": [(f"colite {i}", "p", f"altro {i}") for i in range(10)]}
    index = build([_chunk("c0")], script, max_seeds=3)
    assert len(index._seeds({"colite"})) == 3


# --- idempotence, which the content-hashed chunk ids depend on -------------------------


def test_inserting_the_same_chunks_again_changes_nothing(build):
    chunks = [_chunk("c0"), _chunk("c1")]
    script = {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "ctcae")]}
    index = build(chunks, script)
    before = index._search("steroidi", top_i=10)
    index.insert(chunks)
    assert index._search("steroidi", top_i=10) == before


# --- delete ----------------------------------------------------------------------------


def test_delete_drops_the_sources_reach_and_leaves_the_rest(build):
    index = build(
        [_chunk("c0", "doc1"), _chunk("c1", "doc2")],
        {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "ctcae")]},
    )
    index.delete("doc1")
    assert index._search("steroidi", top_i=10) == []  # only doc1 ever named "steroidi"
    assert set(dict(index._search("ctcae", top_i=10))) == {"c1"}  # doc2 is untouched


def test_delete_of_an_unknown_source_is_a_no_op(build):
    index = build([_chunk("c0")], {"c0": [("colite", "p", "steroidi")]})
    before = index._search("steroidi", top_i=10)
    index.delete("doc-that-never-existed")
    assert index._search("steroidi", top_i=10) == before


def test_delete_then_insert_rebuilds_the_source_cleanly(build):
    """How re-ingestion stays neutral even though the extractor is an LLM: the source is
    rebuilt, not patched, so an answer that changed cannot leave a stale edge behind."""
    chunks = [_chunk("c0", "doc1")]
    index = build(chunks, {"c0": [("colite", "p", "steroidi")]})
    index.delete("doc1")
    index.extractor = _ScriptedExtractor({"c0": [("colite", "p", "budesonide")]})
    index.insert(chunks)
    assert index._search("steroidi", top_i=10) == []
    assert set(dict(index._search("budesonide", top_i=10))) == {"c0"}


# --- access control: the graph does not filter, ``retrieve`` does ----------------------


def test_the_graph_expands_through_chunks_the_subject_cannot_see(build):
    """The accepted cost of filtering only at the end: ``_search`` walks freely, so the
    unauthorized chunk is among the candidates. It is ``retrieve`` that removes it."""
    index = build(
        [
            _chunk("c0", "doc1", {"tenant": "acme"}),
            _chunk("c1", "doc2", {"tenant": "globex"}),
        ],
        {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "ctcae")]},
        schema=_SCHEMA,
    )
    assert set(dict(index._search("steroidi", top_i=10))) == {"c0", "c1"}

    acme = Match(attribute="tenant", values={"acme"})
    assert [sc.chunk.id for sc in index.retrieve("steroidi", top_i=10, filter=acme)] == ["c0"]


def test_an_unlabeled_chunk_is_denied_like_everywhere_else(build):
    index = build(
        [_chunk("c0", "doc1", {"tenant": "acme"}), _chunk("c1", "doc2")],
        {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "ctcae")]},
        schema=_SCHEMA,
    )
    assert [sc.chunk.id for sc in index.retrieve("steroidi", 10, filter=Allow())] == ["c0"]


def test_over_fetch_is_what_keeps_the_filter_from_emptying_the_expansion(build):
    """Nine chunks of another tenant sit on the seed itself; the one chunk this subject
    may read is a hop further out, so it ranks below all of them. With a budget of one and
    no over-fetch it never survives the cut; with over-fetch it does.
    """
    chunks = [_chunk("c0", "doc1", {"tenant": "acme"})] + [
        _chunk(f"x{i}", "doc2", {"tenant": "globex"}) for i in range(9)
    ]
    script = {f"x{i}": [("colite", "p", f"altro {i}")] for i in range(9)}
    script["c0"] = [("altro 0", "p", "steroidi")]  # reached only through "altro 0"
    acme = Match(attribute="tenant", values={"acme"})

    assert build(chunks, script, schema=_SCHEMA, over_fetch=1).retrieve("colite", 1, acme) == []
    generous = build(chunks, script, schema=_SCHEMA, over_fetch=10)
    assert [sc.chunk.id for sc in generous.retrieve("colite", 1, acme)] == ["c0"]


def test_retrieve_never_hands_back_more_than_top_i(build):
    index = build(
        [_chunk(f"c{i}", "doc1", {"tenant": "acme"}) for i in range(10)],
        {f"c{i}": [("colite", "p", f"altro {i}")] for i in range(10)},
        schema=_SCHEMA,
        over_fetch=5,
    )
    assert len(index.retrieve("colite", top_i=3, filter=Allow())) == 3


# --- what only the in-process tier can show ---------------------------------------------


def _volatile(chunks, script, **kwargs):
    store = VolatileStore()
    store.add(chunks)
    index = VolatileGraphIndex(store, _ScriptedExtractor(script), **kwargs)
    index.insert(chunks)
    return index


def test_re_inserting_does_not_grow_the_graph():
    """The parity test above shows the answers do not change; this shows the structure
    does not either, which is the property the content-hashed ids actually rest on."""
    chunks = [_chunk("c0")]
    index = _volatile(chunks, {"c0": [("colite", "p", "steroidi")]})
    before = (index.graph.number_of_nodes(), index.graph.number_of_edges())
    index.insert(chunks)
    assert (index.graph.number_of_nodes(), index.graph.number_of_edges()) == before


def test_two_chunks_stating_the_same_pair_keep_both_provenances():
    """Collapsing them would lose the anchor ``delete`` works by."""
    index = _volatile(
        [_chunk("c0"), _chunk("c1")],
        {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "steroidi")]},
    )
    assert index.graph.number_of_edges() == 2


def test_delete_removes_the_nodes_left_with_nothing():
    index = _volatile(
        [_chunk("c0", "doc1"), _chunk("c1", "doc2")],
        {"c0": [("colite", "p", "steroidi")], "c1": [("colite", "p", "ctcae")]},
    )
    index.delete("doc1")
    assert "steroidi" not in index.graph  # only doc1 ever named it
    assert "colite" in index.graph  # doc2 named it too, so it survives
