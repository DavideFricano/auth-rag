import tempfile
from pathlib import Path

import pytest

from auth_rag.ingestion.converter import BaseConverter
from auth_rag.ingestion.loader import FileLoader, LocalLoader
from auth_rag.types import Document, Origin


class _FakeConverter(BaseConverter):
    """Stands in for MarkdownConverter so the filesystem walk is tested without parsers."""

    def convert_stream(self, data: bytes, media_type: str, name: str = "document") -> str:
        return f"content of {name}"

    def convert_file(self, path: Path) -> str:
        return f"content of {path.name}"


class _FailingConverter(_FakeConverter):
    """Raises on one specific file, to exercise per-item resilience."""

    def convert_file(self, path: Path) -> str:
        if path.name == "bad.txt":
            raise ValueError("boom")
        return super().convert_file(path)


def _loader(input_dir: Path, output_dir: Path, converter=None, **kwargs) -> FileLoader:
    return FileLoader(input_dir, output_dir, converter=converter or _FakeConverter(), **kwargs)


def test_file_loader_is_a_local_loader():
    assert issubclass(FileLoader, LocalLoader)


def test_raises_if_input_dir_missing():
    with pytest.raises(FileNotFoundError):
        _loader(Path("/tmp/cartella_non_esistente_xyz"), Path("/tmp/out"))


def test_raises_if_input_is_file():
    with tempfile.NamedTemporaryFile() as f, pytest.raises(NotADirectoryError):
        _loader(Path(f.name), Path("/tmp/out"))


def test_creates_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "nested" / "output"
        _loader(input_dir, output_dir)
        assert output_dir.exists()


def test_load_returns_documents_for_each_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("a")
        (input_dir / "b.txt").write_text("b")
        (input_dir / "subdir").mkdir()  # non-file entries are skipped

        docs = list(_loader(input_dir, Path(tmpdir) / "out").load())

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert {d.source.name for d in docs} == {"a.txt", "b.txt"}
        # id is the file name (stable identity), not a content hash
        assert {d.source.id for d in docs} == {"a.txt", "b.txt"}
        assert all(d.source.origin == Origin.LOCAL for d in docs)
        assert docs[0].text == "content of a.txt"


def test_load_skips_files_that_fail_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        (input_dir / "good.txt").write_text("g")
        (input_dir / "bad.txt").write_text("b")  # converter raises on this one

        docs = list(_loader(input_dir, Path(tmpdir) / "out", converter=_FailingConverter()).load())

        assert {d.source.name for d in docs} == {"good.txt"}


def test_load_saves_output_when_enabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        (input_dir / "doc.txt").write_text("x")
        output_dir = Path(tmpdir) / "out"

        list(_loader(input_dir, output_dir, save_output=True).load())

        assert (output_dir / "doc.txt.md").read_text() == "content of doc.txt"
