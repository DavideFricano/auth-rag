"""What a caller can catch, and what it must not catch by accident."""

import pytest

from auth_rag.errors import (
    AuthorizationError,
    ConformanceError,
    ConversionError,
    DeclarationError,
    EnforcementError,
    RagError,
)

_AUTHORIZATION = (DeclarationError, ConformanceError, EnforcementError)


@pytest.mark.parametrize("error", [*_AUTHORIZATION, ConversionError])
def test_nothing_is_a_value_error(error):
    """The point of the hierarchy. A backend that wraps a call in `except ValueError` to
    handle bad input must not swallow "this index enforces nothing" along with it: an
    authorization failure has to be caught deliberately, or propagate."""
    assert not issubclass(error, ValueError)
    assert issubclass(error, RagError)


@pytest.mark.parametrize("error", _AUTHORIZATION)
def test_the_three_access_control_cases_share_a_base(error):
    """So a caller that reacts the same way to all of them — refuse to serve, alert — can
    catch one type instead of three."""
    assert issubclass(error, AuthorizationError)


def test_conversion_is_not_an_authorization_failure():
    """A payload nobody can parse is a data problem; the loader is right to skip it. It
    must not be caught by a handler watching for access-control failures."""
    assert not issubclass(ConversionError, AuthorizationError)
