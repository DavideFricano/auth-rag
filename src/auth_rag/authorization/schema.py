from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from auth_rag.authorization.filter import Allow, And, Filter, Match, Not, Or
from auth_rag.errors import ConformanceError, DeclarationError
from auth_rag.types import AttributeValue


class AttributeType(StrEnum):
    """Value type an access attribute accepts; each maps to one exact-match payload index."""

    KEYWORD = "keyword"
    INTEGER = "integer"
    BOOLEAN = "bool"


_PYTHON_TYPE: dict[AttributeType, type] = {
    AttributeType.KEYWORD: str,
    AttributeType.INTEGER: int,
    AttributeType.BOOLEAN: bool,
}


class Attribute(BaseModel):
    """One declared access attribute.

    Unknown keys are refused rather than ignored: the declaration is read from a file, so
    a typo (``requird``) would otherwise leave the attribute optional while its author
    believes it mandatory — an error that fails open.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Payload field name the attribute is indexed under")
    type: AttributeType = Field(description="Value type the attribute accepts")
    multi: bool = Field(default=False, description="Whether a chunk may carry several values")
    required: bool = Field(
        default=False,
        description="Whether every chunk must carry it; one that doesn't is never authorized",
    )


class AccessSchema:
    """The closed vocabulary of access attributes a deployment declares.

    One declaration, three consumers: it validates what the labeler writes, tells each
    index which payload fields to index, and rejects a filter naming an attribute nobody
    declared. Whether one exists is what tells a deployment apart — no schema means no
    ABAC at all; a schema makes labeling and filtering mandatory. A vocabulary that is
    empty, or that requires nothing, is refused: it would claim to do access control while
    obliging no chunk to carry anything.
    """

    def __init__(self, attributes: Sequence[Attribute]) -> None:
        self.attributes = tuple(attributes)
        self._by_name = {attribute.name: attribute for attribute in self.attributes}
        if not self.attributes:
            raise DeclarationError("the access schema declares no attribute")
        if not any(attribute.required for attribute in self.attributes):
            raise DeclarationError(
                "no attribute is required: nothing would then oblige a chunk to carry any, "
                "and an unlabelled one would satisfy a negated predicate"
            )
        if len(self._by_name) != len(self.attributes):
            raise DeclarationError("duplicate attribute name in the access schema")

    @classmethod
    def from_file(cls, path: Path | str) -> AccessSchema:
        """Load the declaration from a JSON list of attributes.

        A file rather than an object because ingestion and query may run as separate
        processes, and both must read the same declaration.
        """
        declared = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(declared, list):
            raise DeclarationError(f"the access schema must be a JSON list of attributes: {path}")
        return cls([Attribute.model_validate(item) for item in declared])

    def is_labeled(self, access: Mapping[str, object]) -> bool:
        """Whether these access attributes carry every attribute the schema requires.

        Checked before the predicate at retrieval, so an unlabeled chunk is denied whatever
        the predicate says — under a bare ``Not`` a missing attribute would satisfy it.
        The read-side twin of ``validate_access``, which raises instead.
        """
        return all(attribute.name in access for attribute in self.attributes if attribute.required)

    def attribute(self, name: str) -> Attribute:
        """The declared attribute under ``name``; an undeclared name never passes."""
        if name not in self._by_name:
            raise ConformanceError(f"undeclared access attribute: {name!r}")
        return self._by_name[name]

    def validate_access(
        self, access: Mapping[str, object]
    ) -> dict[str, AttributeValue | list[AttributeValue]]:
        """Checks what the labeler wrote, before it reaches the store and the payloads.

        Raises on what is wrong *and* on what is missing: without the required attributes
        a document could never be retrieved, so accepting it would only defer the symptom
        to query time as an empty result.
        """
        validated: dict[str, AttributeValue | list[AttributeValue]] = {}
        for name, value in access.items():
            attribute = self.attribute(name)
            if attribute.multi:
                if isinstance(value, str | bytes) or not isinstance(value, Iterable):
                    raise ConformanceError(f"attribute {name!r} is multi-valued: expected a collection")
                validated[name] = [self._checked(attribute, item) for item in value]
            else:
                validated[name] = self._checked(attribute, value)
        if not self.is_labeled(validated):
            missing = [a.name for a in self.attributes if a.required and a.name not in validated]
            raise ConformanceError(f"missing required access attributes: {missing}")
        return validated

    def validate_filter(self, predicate: Filter) -> None:
        """Walks the predicate and rejects any attribute or value the schema doesn't declare."""
        match predicate:
            case Match(attribute=name, values=values):
                attribute = self.attribute(name)
                for value in values:
                    self._checked(attribute, value)
            case And(clauses=clauses) | Or(clauses=clauses):
                for clause in clauses:
                    self.validate_filter(clause)
            case Not(clause=clause):
                self.validate_filter(clause)
            case Allow():
                pass  # no attribute to check: the constant carries no vocabulary
            case _:
                raise ConformanceError(f"unsupported filter node: {type(predicate).__name__}")

    def _checked(self, attribute: Attribute, value: object) -> AttributeValue:
        # bool is an int subclass, so it needs a check of its own in both directions
        expected_bool = attribute.type is AttributeType.BOOLEAN
        if isinstance(value, bool) is not expected_bool or not isinstance(
            value, _PYTHON_TYPE[attribute.type]
        ):
            raise ConformanceError(
                f"attribute {attribute.name!r} expects {attribute.type}, got {value!r}"
            )
        return value
