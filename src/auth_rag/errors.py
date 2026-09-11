from __future__ import annotations


class RagError(Exception):
    """Base of everything this library raises on its own behalf.

    Deliberately not a ``ValueError``: a caller that wraps a call to guard against bad
    input must not swallow an authorization failure by accident. These have to be caught
    on purpose, or not at all.

    Ordinary argument validation — an overlap wider than the chunk, weights that sum to
    zero — stays a plain ``ValueError``, because that is what it is: the caller passed
    something wrong, and nothing about the deployment is at stake.
    """


class ConversionError(RagError):
    """A payload cannot be turned into text: no parser is registered for its media type."""


class AuthorizationError(RagError):
    """Something about access control is wrong. Base of the three cases below, so a caller
    that reacts the same way to all of them can catch just this."""


class DeclarationError(AuthorizationError):
    """A declaration file is unusable: the access schema, or a labeler's manifest.

    Raised while wiring things up, so the honest response is to refuse to start.
    """


class ConformanceError(AuthorizationError):
    """Attributes or a predicate disagree with the declared schema.

    At ingestion it means the producer and the declaration are out of step; at query time
    that the policy names something nobody declared. Either way the deployment needs
    looking at — it is not a bad request.
    """


class EnforcementError(AuthorizationError):
    """This call cannot be enforced as asked.

    Schema and filter go together in both directions: a declared schema with no filter, a
    filter with no schema, or labelled chunks reaching an index that was never given the
    schema. Always a wiring mistake — the data and the code disagree about whether access
    control is on.
    """
