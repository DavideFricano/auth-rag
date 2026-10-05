from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from auth_rag.generation.llm import BaseLLMClient
from auth_rag.types import Chunk, Message

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class Triple(BaseModel):
    """One relation as an extractor read it: two subconcepts and what links them."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(description="Name of the concept the relation starts from")
    predicate: str = Field(description="What relates them; carried, never interpreted")
    object: str = Field(description="Name of the concept the relation reaches")


class BaseExtractor(ABC):
    """Names the concepts a text is about, and how they relate.

    Injected into the relation index the way an embedder is injected into the semantic
    one: it produces that index's representation, so the pipeline never sees it.

    It is asked two different questions, one per side of the index, and that is the point:
    the graph is built and entered by the **same** component, so both sides name concepts
    the same way by construction. Matching a query against node names any other way would
    mean inventing a second vocabulary for the entry, and the entry is where a graph that
    cannot be entered stops being worth building.
    """

    @abstractmethod
    def extract(self, chunk: Chunk) -> list[Triple]:
        """Ingestion side: the relations this chunk states; none of them is a normal answer."""

    @abstractmethod
    def entities(self, text: str) -> list[str]:
        """Query side: the concepts this text is about, without relating them.

        A separate question from ``extract``, because a query usually states no relation
        at all — asked for the relations in "how is colitis treated", an extractor would
        rightly answer none, and the graph would never open.
        """


class LLMExtractor(BaseExtractor):
    """Extraction by an LLM, at temperature 0.

    Determinism is a requirement, not a nicety: chunk ids are content hashes and the store
    upserts on them, so an extractor that answered differently every time would make the
    graph drift from the chunks it points at. Temperature 0 and a pinned prompt hold it;
    ``version`` fingerprints both so two ingestions can be compared.

    A chunk whose extraction fails or comes back unreadable is skipped and logged, like a
    file the loader cannot convert.
    """

    PROMPT = (
        "You extract relations between concepts from a passage.\n"
        "Answer with a JSON array and nothing else. Each element must be an object with "
        'the keys "subject", "predicate" and "object", all short strings.\n'
        "Name concepts as they appear in the passage. Prefer few, specific relations over "
        "many vague ones. If the passage states no relation, answer with []."
    )

    ENTITY_PROMPT = (
        "You name the concepts a question or passage is about.\n"
        "Answer with a JSON array of short strings and nothing else.\n"
        "Name each concept the way it would appear in a document, not as the user phrased "
        "it, and leave out anything that is not a concept. If there is none, answer with []."
    )

    def __init__(
        self,
        llm: BaseLLMClient,
        prompt: str = PROMPT,
        entity_prompt: str = ENTITY_PROMPT,
        max_relations: int = 12,
        max_entities: int = 8,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.entity_prompt = entity_prompt
        self.max_relations = max_relations
        self.max_entities = max_entities

    @property
    def version(self) -> str:
        """Fingerprint of what decides the output: these prompts against this model.

        Identifies a run, not an edge, so it is not stored on the relations. Same version
        and a different graph means the extractor drifted; a different version means it
        was asked to.
        """
        model = getattr(self.llm, "model", type(self.llm).__name__)
        material = f"{model}\n{self.prompt}\n{self.entity_prompt}"
        return f"{model}@{hashlib.sha256(material.encode()).hexdigest()[:12]}"

    def _ask(self, system: str, text: str, what: str) -> object | None:
        """One call at temperature 0, and its answer read back as JSON.

        Returns ``None`` when the model failed or answered something unreadable, which
        both callers treat as "this text yielded nothing" — the loud failure would be the
        wrong one here, since it costs recall and not enforcement.
        """
        messages = [Message(role="system", content=system), Message(role="user", content=text)]
        try:
            answer = self.llm.answer(messages, temperature=0.0)
        except Exception:
            logger.warning("Extraction failed for %s", what, exc_info=True)
            return None
        try:
            return json.loads(_FENCE.sub("", answer).strip())  # models fence their JSON
        except json.JSONDecodeError:
            logger.warning("Extraction for %s was not JSON", what)
            return None

    def entities(self, text: str) -> list[str]:
        parsed = self._ask(self.entity_prompt, text, what="a query")
        if not isinstance(parsed, list):
            if parsed is not None:
                logger.warning("Entity extraction was not a JSON list")
            return []
        named: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip() and item not in named:
                named.append(item)
        return named[: self.max_entities]

    def extract(self, chunk: Chunk) -> list[Triple]:
        parsed = self._ask(self.prompt, chunk.text, what=f"chunk {chunk.id}")
        if not isinstance(parsed, list):
            if parsed is not None:
                logger.warning("Skipping chunk whose extraction was not a JSON list: %s", chunk.id)
            return []
        triples: list[Triple] = []
        for item in parsed:
            # a half-formed element costs that relation, not the chunk: the model
            # naming one end and forgetting the other is its most common slip
            if not isinstance(item, dict):
                continue
            subject, object_ = item.get("subject"), item.get("object")
            if not isinstance(subject, str) or not isinstance(object_, str):
                continue
            predicate = item.get("predicate")
            triple = Triple(
                subject=subject,
                predicate=predicate if isinstance(predicate, str) else "",
                object=object_,
            )
            if triple not in triples:
                triples.append(triple)
        return triples[: self.max_relations]
