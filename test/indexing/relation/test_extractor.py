"""The LLM extractor's parsing and its determinism levers.

The model is faked: what is under test is what the extractor does with an answer, which
is where the failure modes are — a fenced payload, a refusal in prose, a half-formed
object. An unreadable answer must cost one chunk, never the ingestion.
"""

from collections.abc import Iterator
from datetime import date

import pytest

from auth_rag.generation.llm import BaseLLMClient
from auth_rag.indexing.relation.extractor import LLMExtractor, Triple
from auth_rag.types import Chunk, Message, Metadata, Origin, Source


class _FakeLLM(BaseLLMClient):
    """Answers with a canned string and records how it was called."""

    def __init__(self, answer: str = "[]", model: str = "fake-model") -> None:
        self.model = model
        self.answer_text = answer
        self.calls: list[dict] = []

    def answer(self, messages: list[Message], **kwargs) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self.answer_text

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        raise NotImplementedError


class _AngryLLM(BaseLLMClient):
    def answer(self, messages: list[Message], **kwargs) -> str:
        raise RuntimeError("the model is down")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        raise NotImplementedError


def _chunk(text: str = "some text") -> Chunk:
    source = Source(id="doc1", name="d.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id="c0", text=text, metadata=Metadata(source=source, title="S"))


def test_parses_a_json_array():
    llm = _FakeLLM('[{"subject": "A", "predicate": "causes", "object": "B"}]')
    assert LLMExtractor(llm).extract(_chunk()) == [
        Triple(subject="A", predicate="causes", object="B")
    ]


def test_tolerates_the_code_fence_models_wrap_json_in():
    llm = _FakeLLM('```json\n[{"subject": "A", "predicate": "p", "object": "B"}]\n```')
    assert len(LLMExtractor(llm).extract(_chunk())) == 1


def test_an_answer_that_is_not_json_costs_one_chunk_not_the_ingestion():
    """A model that answers in prose is this chunk's problem, like a corrupt file is one
    file's problem — the opposite of a bad access attribute, which stops everything."""
    assert LLMExtractor(_FakeLLM("I cannot help with that.")).extract(_chunk()) == []


def test_an_answer_that_is_not_a_list_is_skipped():
    assert LLMExtractor(_FakeLLM('{"subject": "A"}')).extract(_chunk()) == []


def test_a_model_that_raises_is_skipped():
    assert LLMExtractor(_AngryLLM()).extract(_chunk()) == []


def test_malformed_items_are_dropped_and_the_good_ones_kept():
    llm = _FakeLLM(
        '[{"subject": "A", "object": "B"},'
        ' {"subject": 7, "object": "C"},'
        ' "not an object",'
        ' {"subject": "D", "predicate": "p", "object": "E"}]'
    )
    triples = LLMExtractor(llm).extract(_chunk())
    assert [(t.subject, t.predicate, t.object) for t in triples] == [
        ("A", "", "B"),  # a missing predicate is empty, not a reason to drop the relation
        ("D", "p", "E"),
    ]


def test_the_same_relation_twice_is_one_relation():
    llm = _FakeLLM(
        '[{"subject": "A", "predicate": "p", "object": "B"},'
        ' {"subject": "A", "predicate": "p", "object": "B"}]'
    )
    assert len(LLMExtractor(llm).extract(_chunk())) == 1


def test_max_relations_caps_the_answer():
    items = ",".join(f'{{"subject": "A{i}", "predicate": "p", "object": "B"}}' for i in range(20))
    llm = _FakeLLM(f"[{items}]")
    assert len(LLMExtractor(llm, max_relations=5).extract(_chunk())) == 5


def test_the_call_is_made_at_temperature_zero():
    """The first determinism lever: the same passage must not produce a different graph
    on the next ingestion, or the idempotent upsert of the store loses its meaning."""
    llm = _FakeLLM()
    LLMExtractor(llm).extract(_chunk())
    assert llm.calls[0]["temperature"] == 0.0


def test_the_chunk_text_is_what_the_model_is_asked_about():
    llm = _FakeLLM()
    LLMExtractor(llm).extract(_chunk("ipotiroidismo di grado 1"))
    roles = [m.role for m in llm.calls[0]["messages"]]
    assert roles == ["system", "user"]
    assert llm.calls[0]["messages"][1].content == "ipotiroidismo di grado 1"


@pytest.mark.parametrize(
    "other",
    [
        LLMExtractor(_FakeLLM(model="another-model")),
        LLMExtractor(_FakeLLM(), prompt="a different prompt"),
    ],
)
def test_version_changes_when_either_thing_that_decides_the_output_changes(other):
    """Same version and a different graph means the extractor drifted; a different
    version means it was asked to."""
    assert LLMExtractor(_FakeLLM()).version != other.version


def test_version_is_stable_for_the_same_model_and_prompt():
    assert LLMExtractor(_FakeLLM()).version == LLMExtractor(_FakeLLM()).version


# --- the query side: naming concepts rather than relating them -------------------------


def test_entities_reads_a_json_array_of_names():
    llm = _FakeLLM('["colite immunocorrelata", "steroidi"]')
    assert LLMExtractor(llm).entities("come si tratta la colite?") == [
        "colite immunocorrelata",
        "steroidi",
    ]


def test_entities_is_asked_a_different_question_than_extract():
    """A query states no relation, so asking for relations would answer none and the
    graph would never open. The two prompts are the reason both sides still work."""
    llm = _FakeLLM("[]")
    extractor = LLMExtractor(llm)
    extractor.entities("come si tratta la colite?")
    extractor.extract(_chunk())
    entity_prompt, relation_prompt = (call["messages"][0].content for call in llm.calls)
    assert entity_prompt == LLMExtractor.ENTITY_PROMPT
    assert relation_prompt == LLMExtractor.PROMPT
    assert entity_prompt != relation_prompt


def test_entities_is_asked_at_temperature_zero_too():
    llm = _FakeLLM('["colite"]')
    LLMExtractor(llm).entities("colite")
    assert llm.calls[0]["temperature"] == 0.0


@pytest.mark.parametrize("answer", ["not json at all", '{"concept": "colite"}', "[]"])
def test_entities_returns_nothing_rather_than_failing(answer):
    """No entry point is a normal answer: the similarity indexes still reply, so a query
    the extractor cannot read costs the expansion, not the query."""
    assert LLMExtractor(_FakeLLM(answer)).entities("qualunque cosa") == []


def test_entities_drops_non_strings_and_blanks_and_repeats():
    llm = _FakeLLM('["colite", 7, "   ", "colite", null, "steroidi"]')
    assert LLMExtractor(llm).entities("q") == ["colite", "steroidi"]


def test_max_entities_caps_how_many_doors_one_query_opens():
    names = ",".join(f'"concetto {i}"' for i in range(20))
    assert len(LLMExtractor(_FakeLLM(f"[{names}]"), max_entities=3).entities("q")) == 3


def test_a_model_that_raises_yields_no_entities():
    assert LLMExtractor(_AngryLLM()).entities("colite") == []


def test_version_changes_when_the_entity_prompt_changes():
    """It fingerprints everything that decides the graph, and the entry is part of it."""
    other = LLMExtractor(_FakeLLM(), entity_prompt="a different entity prompt")
    assert LLMExtractor(_FakeLLM()).version != other.version
