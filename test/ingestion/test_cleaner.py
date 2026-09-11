from auth_rag.ingestion.cleaner import Cleaner


def test_collapses_whitespace():
    assert Cleaner.make_plain("foo   bar") == "foo bar"


def test_strips_edges():
    assert Cleaner.make_plain("  hello  ") == "hello"


def test_newlines_and_tabs():
    assert Cleaner.make_plain("a\n\tb") == "a b"


def test_collapses_excess_newlines():
    assert Cleaner.normalize_newlines("a\n\n\n\nb") == "a\n\nb"


def test_preserves_paragraph_breaks():
    assert Cleaner.normalize_newlines("a\n\nb\nc") == "a\n\nb\nc"


def test_strips_trailing_whitespace_per_line():
    assert Cleaner.normalize_newlines("a  \nb\t\n") == "a\nb"


def test_removes_control_chars():
    assert Cleaner.remove_control_chars("a\x00b\x0cc\x7fd") == "abcd"


def test_keeps_newlines_and_tabs():
    assert Cleaner.remove_control_chars("a\nb\tc") == "a\nb\tc"
