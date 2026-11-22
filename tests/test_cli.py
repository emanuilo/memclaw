"""Tests for the `memclaw` command: each command reaches the code that does
it, and stops with a clear message when a setting it needs is missing."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from memclaw import backends
from memclaw.bot import Platform
from memclaw.cli import cli


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, isolate_claude_env):
    """Keep the developer's own settings out of the commands."""
    for name in ("AGENT_BACKEND", "CURSOR_MODEL", "CURSOR_API_KEY", "MEMCLAW_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def _run(tmp_path: Path, *args: str, exit_code: int = 0) -> str:
    result = CliRunner().invoke(cli, ["--memory-dir", str(tmp_path / "m"), *args], obj={})
    assert result.exit_code == exit_code, result.output
    return result.output


class _OneShotBackend:
    """A registered backend that only answers consolidation's one-shot call."""

    name = "fake"

    def __init__(self, config):
        pass

    @classmethod
    def is_configured(cls, config):
        return True

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        return "## Key Facts\n- Consolidated"


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setitem(backends.REGISTRY, "fake", _OneShotBackend)
    monkeypatch.setenv("AGENT_BACKEND", "fake")


# ── status ──────────────────────────────────────────────────────────


def test_claude_backend_shows_its_model_and_effort(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    output = _run(tmp_path, "status")

    assert "Model            : claude-opus-5" in output
    assert "Effort           : high" in output


def test_cursor_backend_shows_only_its_own_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "cursor")
    monkeypatch.setenv("CURSOR_MODEL", "composer-3")
    monkeypatch.delenv("CURSOR_EFFORT", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    output = _run(tmp_path, "status")

    assert "Model            : composer-3" in output
    assert "Effort           : default" in output
    assert "claude-opus-5" not in output


def test_general_rows_are_still_shown(tmp_path: Path):
    output = _run(tmp_path, "status")

    labels = ["Memory directory", "Memory files", "Platform", "Model",
              "Indexed chunks", "Stored images", "Database"]
    positions = [output.index(label) for label in labels]
    assert positions == sorted(positions)
    assert "Platform         : terminal" in output


# ── save, search, index ─────────────────────────────────────────────


def test_save_then_search_then_index(tmp_path: Path):
    assert "Memory saved to" in _run(tmp_path, "save", "The spare key is under the blue flowerpot")

    output = _run(tmp_path, "search", "spare key flowerpot", "--limit", "1")
    assert "[1]" in output and "blue flowerpot" in output

    assert "Index updated" in _run(tmp_path, "index")  # AGENTS.md, MEMORY.md
    output = _run(tmp_path, "index")
    assert "Index already up to date" in output
    assert "Chunks: 2  Files: 2" in output


def test_search_with_nothing_saved(tmp_path: Path):
    assert "No matching memories found." in _run(tmp_path, "search", "anything")


# ── consolidate ─────────────────────────────────────────────────────


def test_consolidate_runs_with_the_chosen_backend(tmp_path: Path, fake_backend):
    memory = tmp_path / "m"
    (memory / "memory").mkdir(parents=True)
    yesterday = date.today() - timedelta(days=1)
    (memory / "memory" / f"{yesterday.isoformat()}.md").write_text("# Yesterday\n\nA fact\n")

    output = _run(tmp_path, "consolidate")

    assert "Consolidation complete." in output
    assert "- Consolidated" in (memory / "MEMORY.md").read_text()


def test_consolidate_with_nothing_to_do(tmp_path: Path, fake_backend):
    assert "No daily files to consolidate." in _run(tmp_path, "consolidate")


def test_consolidate_rejects_a_bad_date(tmp_path: Path, fake_backend):
    output = _run(tmp_path, "consolidate", "--since", "yesterday", exit_code=1)
    assert "Invalid date format: yesterday. Use YYYY-MM-DD." in output


def test_an_unknown_backend_is_named(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "nope")

    output = _run(tmp_path, "consolidate", exit_code=1)

    assert "Error: unknown agent backend nope (set via AGENT_BACKEND)." in output
    assert "Available: claude, cursor." in " ".join(output.split())


def test_a_backend_without_credentials_says_what_to_set(tmp_path: Path):
    output = _run(tmp_path, "consolidate", exit_code=1)

    assert "Error: no Claude credential is configured." in output
    assert "Run memclaw configure to set it." in output


# ── bare `memclaw` ──────────────────────────────────────────────────


@pytest.mark.parametrize("configured, started", [("", "terminal"), ("slack", "slack")])
def test_bare_memclaw_starts_the_chosen_platform(
    tmp_path: Path, monkeypatch, fake_backend, configured, started,
):
    monkeypatch.setenv("MEMCLAW_PLATFORM", configured)
    monkeypatch.setattr("memclaw.cli.needs_setup", lambda: False)
    ran: list[str] = []
    monkeypatch.setattr(Platform, "run", lambda self, config: ran.append(self.name))

    _run(tmp_path)

    assert ran == [started]
