from pathlib import Path

import pytest

from auth_rag.config import Settings


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Settings sees only code defaults: no .env in the cwd, no ambient env vars.

    A defaults test must not depend on the developer's .env — that would test the
    machine, not the code. env var names are derived from the model fields.
    """
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_defaults(isolated_env):
    s = Settings()
    assert s.llm_url == "http://localhost:11434/api/chat"
    assert s.top_k == 10
    assert s.in_dir == Path("data/raw")


def test_env_override(isolated_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    s = Settings()
    assert s.llm_model == "custom-model"
