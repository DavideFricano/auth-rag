"""The seeders: how a query is read, and how it is matched against the concept names.

The graph is stood in for by a stub that records what it was asked to match, so these
check the seeder's own decisions; the tiers' matching is covered in test_graph_index.py.
"""

import numpy as np
import pytest
from fastembed import SparseTextEmbedding
from fastembed.common.utils import remove_non_alphanumeric

from auth_rag.embedding.embedder import BaseEmbedder
from auth_rag.indexing.relation.extractor import BaseExtractor, Triple
from auth_rag.indexing.relation.seeder import LexicalSeeder, SemanticSeeder, bm25, roots
from auth_rag.types import Chunk, Language


class _Graph:
    """Answers every match with preset scores, and records the roots it was asked."""

    def __init__(self, scores: dict[str, float], max_seeds: int = 10) -> None:
        self.scores = scores
        self.max_seeds = max_seeds
        self.asked: list[list[str]] = []

    def _match(self, roots_: list[str]) -> dict[str, float]:
        self.asked.append(roots_)
        return self.scores.get(" ".join(roots_), {})

    def _nearest(self, vector: list[float]) -> dict[str, float]:
        self.asked.append(vector)
        return self.scores.get(tuple(vector), {})


class _NamingExtractor(BaseExtractor):
    def __init__(self, named: list[str]) -> None:
        self.named = named

    def extract(self, chunk: Chunk) -> list[Triple]:
        return []

    def entities(self, text: str) -> list[str]:
        return self.named


# --- roots: the analysis, which has to be the LexicalIndex's -----------------------------


def test_roots_drop_stopwords_and_fold_inflections():
    assert roots("Come si curano le coliti?", Language.ITALIAN) == ["cur", "col"]
    assert roots("la colite", Language.ITALIAN) == ["col"]


_SAMPLES = {
    Language.ENGLISH: "How are the patients' colitis treatments going, 3 days in?",
    Language.ITALIAN: "Effetti collaterali dell'immunoterapia nei pazienti anziani, grado 3",
    Language.FRENCH: "Les traitements des colites chez les patients âgés",
    Language.GERMAN: "Die Behandlungen der Kolitis bei älteren Patienten",
    Language.SPANISH: "Los tratamientos de las colitis en pacientes mayores",
}


@pytest.mark.parametrize("language", list(Language))
def test_roots_read_text_exactly_as_fastembed_does(language):
    """The contract the readable roots rest on: if FastEmbed changed its analysis, the graph
    and the LexicalIndex would read text differently without anything failing but this."""
    model = SparseTextEmbedding("Qdrant/bm25", language=str(language)).model
    text = _SAMPLES[language]
    expected = model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text)))
    assert roots(text, language) == expected


# --- bm25 ------------------------------------------------------------------------------


def test_bm25_scores_only_what_shares_a_root():
    documents = {"colite": ["col"], "budesonide": ["budesonid"]}
    assert set(bm25(documents, ["col"])) == {"colite"}


def test_bm25_weighs_a_rare_root_above_a_common_one():
    documents = {"a": ["col", "acut"], "b": ["col", "cronic"], "c": ["col"]}
    scores = bm25(documents, ["col", "cronic"])
    assert max(scores, key=scores.get) == "b"


def test_bm25_of_nothing_is_nothing():
    assert bm25({}, ["col"]) == {}
    assert bm25({"colite": ["col"]}, []) == {}


# --- the lexical seeder ----------------------------------------------------------------


def test_without_an_extractor_the_query_is_matched_as_it_is():
    graph = _Graph({"cur col": {"colite": 2.0}})
    seeds = LexicalSeeder(language=Language.ITALIAN).seeds("come si curano le coliti?", graph)
    assert graph.asked == [["cur", "col"]]
    assert seeds == {"colite": 1.0}


def test_with_an_extractor_each_concept_it_names_is_matched_on_its_own():
    """One match per concept: joined, they would be a long text against short names again."""
    graph = _Graph(
        {"col immunocorrel": {"colite immunocorrelata": 3.0}, "steroid": {"steroidi": 1.5}}
    )
    extractor = _NamingExtractor(["colite immunocorrelata", "steroidi"])
    seeds = LexicalSeeder(extractor, Language.ITALIAN).seeds("qual è la cura?", graph)
    assert graph.asked == [["col", "immunocorrel"], ["steroid"]]
    assert seeds == {"colite immunocorrelata": 1.0, "steroidi": 0.5}


def test_a_concept_found_twice_keeps_its_best_score():
    graph = _Graph({"col": {"colite": 1.0}, "col acut": {"colite": 4.0, "acuta": 2.0}})
    extractor = _NamingExtractor(["colite", "colite acuta"])
    seeds = LexicalSeeder(extractor, Language.ITALIAN).seeds("q", graph)
    assert seeds == {"colite": 1.0, "acuta": 0.5}


def test_an_extractor_that_names_nothing_opens_nothing():
    graph = _Graph({})
    assert LexicalSeeder(_NamingExtractor([]), Language.ITALIAN).seeds("q", graph) == {}
    assert graph.asked == []


def test_max_seeds_caps_the_entry_points():
    graph = _Graph({"col": {f"colite {i}": float(i + 1) for i in range(5)}}, max_seeds=2)
    seeds = LexicalSeeder(language=Language.ITALIAN).seeds("colite", graph)
    assert list(seeds) == ["colite 4", "colite 3"]


def test_the_seeder_hands_the_node_its_roots():
    seeder = LexicalSeeder(language=Language.ITALIAN)
    assert seeder.represent("Coliti acute") == {"roots": "col acut"}


# --- the semantic seeder ---------------------------------------------------------------


class _Embedder(BaseEmbedder):
    """One axis per text, so a vector says which text it came from; records who asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.0] if "colite" in text else [0.0, 1.0]

    def embed_chunks(self, chunks):
        self.calls.append(("chunks", list(chunks)))
        return np.array([self._vector(text) for text in chunks])

    def embed_query(self, query):
        self.calls.append(("query", [query]))
        return np.array([self._vector(query)])


def test_without_an_extractor_the_query_is_embedded_as_a_query():
    embedder = _Embedder()
    graph = _Graph({(1.0, 0.0): {"colite": 0.8, "colite acuta": 0.4}})
    seeds = SemanticSeeder(embedder).seeds("come si cura la colite?", graph)
    assert embedder.calls == [("query", ["come si cura la colite?"])]
    assert seeds == {"colite": 1.0, "colite acuta": 0.5}


def test_with_an_extractor_each_concept_is_embedded_and_matched_on_its_own():
    embedder = _Embedder()
    graph = _Graph({(1.0, 0.0): {"colite": 0.9}, (0.0, 1.0): {"steroidi": 0.6}})
    extractor = _NamingExtractor(["colite", "steroidi"])
    seeds = SemanticSeeder(embedder, extractor).seeds("q", graph)
    assert embedder.calls == [("chunks", ["colite", "steroidi"])]
    assert graph.asked == [[1.0, 0.0], [0.0, 1.0]]
    assert seeds == {"colite": 1.0, "steroidi": pytest.approx(0.6 / 0.9)}


def test_a_concept_not_similar_enough_is_not_entered():
    graph = _Graph({(1.0, 0.0): {"colite": 0.8, "glicemia": 0.2}})
    seeds = SemanticSeeder(_Embedder(), min_similarity=0.5).seeds("colite", graph)
    assert seeds == {"colite": 1.0}


def test_the_semantic_seeder_hands_the_node_its_embedding():
    assert SemanticSeeder(_Embedder()).represent("colite") == {"embedding": [1.0, 0.0]}

