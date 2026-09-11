from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from auth_rag.errors import ConformanceError
from auth_rag.types import AttributeValue


class Filter(BaseModel):
    """Backend-neutral authorization predicate: what the PEP hands to the indexes.

    Neither the policy engine's dialect nor the backend's: each index translates this into
    its own. Nodes are built by keyword — ``And(clauses=[a, b])`` — are immutable, and
    refuse unknown keys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Match(Filter):
    """The chunk's ``attribute`` holds one of ``values``; a single value is equality.

    No values is the canonical deny: it matches nothing.
    """

    type: Literal["match"] = "match"
    attribute: str = Field(description="Name of the access attribute to test")
    values: frozenset[AttributeValue] = Field(description="Values authorized for it")


class And(Filter):
    """Every clause must hold. At least one is required: an empty conjunction would be
    vacuously true, and so authorize everything."""

    type: Literal["and"] = "and"
    clauses: tuple[Clause, ...] = Field(min_length=1, description="Clauses to conjoin")


class Or(Filter):
    """At least one clause must hold, and at least one must be given."""

    type: Literal["or"] = "or"
    clauses: tuple[Clause, ...] = Field(min_length=1, description="Clauses to disjoin")


class Not(Filter):
    """The clause must not hold."""

    type: Literal["not"] = "not"
    clause: Clause = Field(description="Clause to negate")


class Allow(Filter):
    """Everything is authorized: the explicit spelling of "no restriction".

    Where a schema is declared the filter is mandatory, so this is how a call that has no
    restriction says so rather than omitting the argument. It waives the policy only: a
    chunk missing an attribute the schema requires stays out even under ``Allow()``.
    """

    type: Literal["allow"] = "allow"


Clause = Annotated[Match | And | Or | Not | Allow, Field(discriminator="type")]
"""Any node of the algebra, tagged so a serialized tree can be read back as itself.

Defined after the nodes because it names them while they name it; their clause fields are
forward references pydantic resolves on first use.
"""

FilterAdapter: TypeAdapter[Clause] = TypeAdapter(Clause)
"""Reads a predicate back from JSON or a dict. Writing one needs nothing special:
``predicate.model_dump_json()``."""


def evaluate(
    predicate: Filter, access: Mapping[str, AttributeValue | list[AttributeValue]]
) -> bool:
    """Whether these access attributes satisfy the predicate.

    The reference semantics of the algebra, and what any backend pushdown must agree with.
    Two behaviours callers depend on: an attribute the chunk doesn't carry never matches,
    and values are compared by type as well, so ``True`` does not satisfy a filter asking
    for ``1``.
    """
    match predicate:
        case Match(attribute=name, values=values):
            if name not in access:
                return False
            held = access[name]
            candidates = held if isinstance(held, list) else [held]
            return any(
                type(value) is type(candidate) and value == candidate
                for candidate in candidates
                for value in values
            )
        case And(clauses=clauses):
            return all(evaluate(clause, access) for clause in clauses)
        case Or(clauses=clauses):
            return any(evaluate(clause, access) for clause in clauses)
        case Not(clause=clause):
            return not evaluate(clause, access)
        case Allow():
            return True
        case _:
            raise ConformanceError(f"unsupported filter node: {type(predicate).__name__}")
