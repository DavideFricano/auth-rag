import pytest
from pydantic import ValidationError

from auth_rag.authorization.filter import (
    Allow,
    And,
    Filter,
    FilterAdapter,
    Match,
    Not,
    Or,
    evaluate,
)
from auth_rag.errors import ConformanceError


def test_nested_clauses_keep_their_concrete_type():
    """Clauses are declared as the abstract Filter, so the tree must not be flattened to
    it — an index translating the predicate needs to see Match, And, Or, Not."""
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Not(clause=Match(attribute="classification", values={"confidential"})),
    ])
    assert [type(clause).__name__ for clause in predicate.clauses] == ["Match", "Not"]


def test_filters_are_frozen_and_hashable():
    predicate = Match(attribute="tenant", values={"acme"})
    assert hash(predicate)
    with pytest.raises(ValidationError):
        predicate.attribute = "other"


def test_empty_conjunction_is_refused():
    """An empty AND is vacuously true: allowed through, it would authorize the whole
    corpus without raising anything."""
    with pytest.raises(ValidationError):
        And()


def test_empty_disjunction_is_refused():
    """Safe on its own (an empty OR denies everything) but refused anyway, so a deny has
    exactly one spelling: a Match with no values."""
    with pytest.raises(ValidationError):
        Or()


def test_match_with_no_values_is_the_canonical_deny():
    assert Match(attribute="tenant", values=set()).values == frozenset()


def test_booleans_survive_the_value_union():
    """str | int | bool must not coerce True into 1, or a bool attribute stops matching."""
    values = Match(attribute="exportable", values={True}).values
    assert type(next(iter(values))) is bool


def test_match_on_a_scalar_attribute():
    access = {"tenant": "acme"}
    assert evaluate(Match(attribute="tenant", values={"acme", "globex"}), access) is True
    assert evaluate(Match(attribute="tenant", values={"globex"}), access) is False


def test_missing_attribute_denies():
    """Default-deny: a chunk the labeler never tagged must not satisfy a filter, or every
    unlabeled chunk becomes readable by everyone."""
    assert evaluate(Match(attribute="tenant", values={"acme"}), {}) is False
    assert evaluate(Match(attribute="tenant", values={"acme"}), {"other": "acme"}) is False


def test_multi_valued_attribute_matches_on_overlap():
    """A chunk carrying several values for one attribute matches if any of them is
    authorized — the same semantics as MatchAny against an array payload."""
    access = {"care_team": ["cardiology", "icu"]}
    assert evaluate(Match(attribute="care_team", values={"icu"}), access) is True
    assert evaluate(Match(attribute="care_team", values={"oncology"}), access) is False


def test_match_with_no_values_never_matches():
    """The canonical deny: a policy that authorized no value for the attribute."""
    assert evaluate(Match(attribute="tenant", values=set()), {"tenant": "acme"}) is False


def test_bool_and_int_do_not_satisfy_each_other():
    """bool subclasses int and True == 1, so a plain membership test would let a filter
    asking for 1 be satisfied by an attribute holding True."""
    assert evaluate(Match(attribute="exportable", values={1}), {"exportable": True}) is False
    assert evaluate(Match(attribute="retention_years", values={True}), {"retention_years": 1}) is False
    assert evaluate(Match(attribute="exportable", values={True}), {"exportable": True}) is True


def test_and_or_not():
    access = {"tenant": "acme", "classification": "confidential"}
    tenant = Match(attribute="tenant", values={"acme"})
    secret = Match(attribute="classification", values={"confidential"})

    assert evaluate(And(clauses=[tenant, secret]), access) is True
    assert evaluate(And(clauses=[tenant, Not(clause=secret)]), access) is False
    assert evaluate(Or(clauses=[Not(clause=tenant), secret]), access) is True
    assert evaluate(Not(clause=Not(clause=tenant)), access) is True


def test_nested_tree_is_walked_fully():
    access = {"tenant": "acme", "classification": "public", "care_team": ["icu"]}
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Or(clauses=[
            Match(attribute="classification", values={"public", "internal"}),
            Match(attribute="care_team", values={"cardiology"}),
        ]),
        Not(clause=Match(attribute="classification", values={"confidential"})),
    ])
    assert evaluate(predicate, access) is True


def test_allow_is_the_constant_true():
    """The counterpart of the canonical deny: it holds whatever the chunk carries, so a
    caller that has no restriction can say so instead of omitting the argument."""
    assert evaluate(Allow(), {}) is True
    assert evaluate(Allow(), {"tenant": "acme"}) is True
    assert evaluate(Not(clause=Allow()), {"tenant": "acme"}) is False


def test_allow_takes_no_argument():
    """It is the constant of the algebra: no attribute, no values, nothing to configure —
    which is also why two of them are interchangeable."""
    assert Allow() == Allow()
    with pytest.raises(ValidationError):
        Allow(attribute="tenant")


def test_unknown_node_raises_instead_of_defaulting():
    """A node type the evaluator doesn't handle must not silently evaluate to True or
    False — either default would be a wrong authorization decision."""

    class _Unhandled(Filter):
        pass

    with pytest.raises(ConformanceError, match="unsupported"):
        evaluate(_Unhandled(), {"tenant": "acme"})


def test_a_predicate_survives_a_round_trip():
    """The reason every node carries a type tag. Before it, clauses declared as the
    fieldless base serialized to `{}` — without raising — so an audit log would have
    recorded a predicate that means nothing, and a filter crossing a process boundary
    would have arrived empty."""
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Not(clause=Match(attribute="classification", values={"confidential"})),
        Or(clauses=[Allow(), Match(attribute="care_team", values={"icu"})]),
    ])

    back = FilterAdapter.validate_json(predicate.model_dump_json())

    assert back == predicate
    assert [type(clause).__name__ for clause in back.clauses] == ["Match", "Not", "Or"]


def test_the_serialized_form_keeps_the_attributes_and_values():
    """What an audit log actually needs to read back."""
    dumped = Match(attribute="tenant", values={"acme"}).model_dump()
    assert dumped == {"type": "match", "attribute": "tenant", "values": frozenset({"acme"})}


def test_a_round_tripped_predicate_decides_the_same_way():
    """Equality is not enough on its own: what matters is that the rebuilt tree authorizes
    exactly what the original did."""
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme", "globex"}),
        Not(clause=Match(attribute="exportable", values={True})),
    ])
    back = FilterAdapter.validate_json(predicate.model_dump_json())

    for access in (
        {"tenant": "acme", "exportable": False},
        {"tenant": "acme", "exportable": True},
        {"tenant": "other", "exportable": False},
        {},
    ):
        assert evaluate(back, access) == evaluate(predicate, access)


def test_an_unknown_node_type_is_refused_on_parse():
    """A tag nobody declared must not be read as a bare Filter that evaluate would then
    reject at decision time — it fails at the boundary instead."""
    with pytest.raises(ValidationError):
        FilterAdapter.validate_json('{"type": "maybe", "attribute": "tenant"}')
