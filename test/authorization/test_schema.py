import json

import pytest
from pydantic import ValidationError

from auth_rag.authorization.filter import Allow, And, Match, Not, Or
from auth_rag.authorization.schema import AccessSchema, Attribute, AttributeType
from auth_rag.errors import ConformanceError, DeclarationError


def _schema() -> AccessSchema:
    return AccessSchema(
        [
            Attribute(name="tenant", type=AttributeType.KEYWORD, required=True),
            Attribute(name="classification", type=AttributeType.KEYWORD),
            Attribute(name="care_team", type=AttributeType.KEYWORD, multi=True),
            Attribute(name="retention_years", type=AttributeType.INTEGER),
            Attribute(name="exportable", type=AttributeType.BOOLEAN),
        ]
    )


def _write(tmp_path, text: str) -> str:
    """Write the declaration verbatim, as a person editing the file would — so the tests
    below pin the on-disk format itself, not a round-trip through ``json.dumps``."""
    path = tmp_path / "access_schema.json"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_declaration_is_loaded_from_a_json_file(tmp_path):
    """Ingestion and query may be separate processes, so what makes the contract hold is
    that both read the same declaration — not that they share an object."""
    path = _write(
        tmp_path,
        """
        [
          { "name": "tenant", "type": "keyword", "required": true },
          { "name": "care_team", "type": "keyword", "multi": true }
        ]
        """,
    )
    schema = AccessSchema.from_file(path)
    assert [attribute.name for attribute in schema.attributes] == ["tenant", "care_team"]
    assert schema.attribute("tenant").required is True
    assert schema.attribute("care_team").multi is True


def test_a_typo_in_the_declaration_is_refused(tmp_path):
    """The reason unknown keys are forbidden: 'requird' would be dropped in silence and
    leave the attribute optional while its author believes it mandatory — failing open."""
    path = _write(tmp_path, '[{ "name": "tenant", "type": "keyword", "requird": true }]')
    with pytest.raises(ValidationError):
        AccessSchema.from_file(path)


def test_attributes_are_not_required_unless_declared_so():
    assert Attribute(name="tenant", type=AttributeType.KEYWORD).required is False


def test_a_declaration_that_is_not_a_list_is_refused(tmp_path):
    path = _write(tmp_path, '{ "attributes": [{ "name": "tenant", "type": "keyword" }] }')
    with pytest.raises(DeclarationError, match="JSON list"):
        AccessSchema.from_file(path)


def test_a_malformed_file_fails_at_load_time(tmp_path):
    """The file is hand-edited, so a stray comma is a realistic way to break it. It must
    stop the process at startup, not leave a half-built vocabulary behind."""
    path = _write(tmp_path, '[{ "name": "tenant", "type": "keyword" },]')
    with pytest.raises(json.JSONDecodeError):
        AccessSchema.from_file(path)


def test_a_vocabulary_that_requires_nothing_is_refused():
    """The same half-configuration in another guise: with no required attribute nothing
    obliges a chunk to carry any, so ``is_labeled`` always passes and a bare negation lets
    an unlabelled chunk through — a schema that looks like access control and isn't."""
    with pytest.raises(DeclarationError, match="no attribute is required"):
        AccessSchema([Attribute(name="tenant", type=AttributeType.KEYWORD)])


def test_an_empty_vocabulary_is_refused():
    """It would claim the deployment does ABAC — filter mandatory — while making every
    filter invalid, since no attribute name would be declared."""
    with pytest.raises(DeclarationError, match="no attribute"):
        AccessSchema([])


def test_is_labeled_asks_only_for_the_required_attributes():
    schema = AccessSchema(
        [
            Attribute(name="tenant", type=AttributeType.KEYWORD, required=True),
            Attribute(name="classification", type=AttributeType.KEYWORD),
        ]
    )
    assert schema.is_labeled({"tenant": "acme"}) is True
    assert schema.is_labeled({"tenant": "acme", "classification": "public"}) is True
    assert schema.is_labeled({"classification": "public"}) is False
    assert schema.is_labeled({}) is False


def test_validate_access_refuses_what_is_missing_not_only_what_is_wrong():
    """Completeness belongs here rather than in the labeler: the schema owns what
    `required` means, so every writer inherits the check instead of remembering it."""
    schema = AccessSchema(
        [
            Attribute(name="tenant", type=AttributeType.KEYWORD, required=True),
            Attribute(name="classification", type=AttributeType.KEYWORD),
        ]
    )
    assert schema.validate_access({"tenant": "acme"}) == {"tenant": "acme"}
    with pytest.raises(ConformanceError, match=r"missing required.*tenant"):
        schema.validate_access({"classification": "public"})
    with pytest.raises(ConformanceError, match="missing required"):
        schema.validate_access({})


def test_validate_filter_accepts_the_constant():
    """Allow names no attribute, so there is no vocabulary to check it against."""
    assert _schema().validate_filter(Allow()) is None
    predicate = And(clauses=[Allow(), Match(attribute="tenant", values={"acme"})])
    assert _schema().validate_filter(predicate) is None


def test_duplicate_declaration_is_refused():
    with pytest.raises(DeclarationError):
        AccessSchema(
            [
                Attribute(name="tenant", type=AttributeType.KEYWORD),
                Attribute(name="tenant", type=AttributeType.INTEGER),
            ]
        )


def test_validate_access_accepts_declared_attributes():
    access = _schema().validate_access(
        {
            "tenant": "acme",
            "care_team": ["cardiology", "icu"],
            "retention_years": 10,
            "exportable": False,
        }
    )
    assert access["care_team"] == ["cardiology", "icu"]
    assert access["exportable"] is False


def test_validate_access_rejects_undeclared_attribute():
    """A typo must fail at ingestion, not become an attribute nothing ever filters on."""
    with pytest.raises(ConformanceError, match="undeclared"):
        _schema().validate_access({"tenat": "acme"})


def test_validate_access_rejects_wrong_type():
    with pytest.raises(ConformanceError, match="expects"):
        _schema().validate_access({"retention_years": "ten"})


def test_bool_and_int_are_not_interchangeable():
    """bool subclasses int in Python, so without an explicit check 0 would pass as a
    boolean and True as an integer."""
    with pytest.raises(ConformanceError):
        _schema().validate_access({"exportable": 0})
    with pytest.raises(ConformanceError):
        _schema().validate_access({"retention_years": True})


def test_multi_valued_attribute_wants_a_collection():
    with pytest.raises(ConformanceError, match="multi-valued"):
        _schema().validate_access({"care_team": "cardiology"})


def test_single_valued_attribute_refuses_a_collection():
    with pytest.raises(ConformanceError, match="expects"):
        _schema().validate_access({"tenant": ["acme", "globex"]})


def test_validate_filter_walks_the_whole_tree():
    """The undeclared attribute is buried under And -> Or -> Not, so a shallow check
    would let it reach the backend and silently match nothing."""
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Or(clauses=[
            Match(attribute="classification", values={"public"}),
            Not(clause=Match(attribute="undeclared", values={"x"})),
        ]),
    ])
    with pytest.raises(ConformanceError, match="undeclared"):
        _schema().validate_filter(predicate)


def test_validate_filter_accepts_a_well_formed_predicate():
    predicate = And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Not(clause=Match(attribute="classification", values={"confidential"})),
    ])
    assert _schema().validate_filter(predicate) is None


def test_validate_filter_checks_value_types_too():
    with pytest.raises(ConformanceError, match="expects"):
        _schema().validate_filter(Match(attribute="retention_years", values={"ten"}))
