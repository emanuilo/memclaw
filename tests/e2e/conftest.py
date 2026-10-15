"""End-to-end memory validation against the real APIs.

These tests use real OpenAI embeddings and (for the agent tests) a real
Claude backend, each against a throwaway vault in a temp directory — your
own ~/.memclaw is never touched. They are skipped unless MEMCLAW_E2E=1:

    MEMCLAW_E2E=1 pytest tests/e2e -v

Keys come from the environment, a ./.env or ~/.memclaw/.env (the same
places memclaw itself reads). MEMCLAW_E2E_MODEL picks the Claude model for
the agent tests (defaults to claude-haiku-5-5 to keep runs cheap).
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from memclaw.config import MemclawConfig

E2E_ENABLED = os.environ.get("MEMCLAW_E2E") == "1"
E2E_MODEL = os.environ.get("MEMCLAW_E2E_MODEL", "claude-haiku-5-5")


def pytest_collection_modifyitems(config, items):
    if E2E_ENABLED:
        return
    skip = pytest.mark.skip(reason="end-to-end tests; set MEMCLAW_E2E=1 to run")
    for item in items:
        if "tests/e2e/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(skip)


def _require(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"missing {', '.join(missing)}")


@pytest.fixture
def vault(tmp_path: Path) -> MemclawConfig:
    """A fresh vault with real OpenAI credentials."""
    _require("OPENAI_API_KEY")
    return MemclawConfig(memory_dir=tmp_path / "vault")


@pytest.fixture
def agent_vault(tmp_path: Path, monkeypatch) -> MemclawConfig:
    """A fresh vault configured for the real Claude backend."""
    _require("OPENAI_API_KEY")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        pytest.skip("missing ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN")
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)
    return MemclawConfig(
        memory_dir=tmp_path / "vault",
        agent_backend="claude",
        claude_model=E2E_MODEL,
    )


def days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


def write_daily(config: MemclawConfig, day: date, entries: list[tuple[str, str]]) -> Path:
    """Write a daily file in the format MemoryStore.save produces."""
    lines = [f"# {day.strftime('%A, %B %d, %Y')}", ""]
    for hhmm, text in entries:
        lines += ["", f"## {hhmm} - Note", "", text, "", "---"]
    path = config.daily_file(day)
    path.write_text("\n".join(lines) + "\n")
    return path
