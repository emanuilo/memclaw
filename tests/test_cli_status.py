"""Tests for `memclaw status`."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memclaw.cli import cli


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, isolate_claude_env):
    """Keep the developer's own backend settings out of the status output."""
    for name in ("AGENT_BACKEND", "CURSOR_MODEL", "MEMCLAW_PLATFORM"):
        monkeypatch.delenv(name, raising=False)


def _status(tmp_path: Path) -> str:
    result = CliRunner().invoke(
        cli, ["--memory-dir", str(tmp_path / "m"), "status"], obj={},
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_claude_backend_shows_its_model_and_effort(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    output = _status(tmp_path)

    assert "Model            : claude-opus-5" in output
    assert "Effort           : high" in output


def test_cursor_backend_shows_only_its_own_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "cursor")
    monkeypatch.setenv("CURSOR_MODEL", "composer-3")
    monkeypatch.delenv("CURSOR_EFFORT", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")

    output = _status(tmp_path)

    assert "Model            : composer-3" in output
    assert "Effort           : default" in output
    assert "claude-opus-5" not in output


def test_general_rows_are_still_shown(tmp_path: Path):
    output = _status(tmp_path)

    for label in ("Memory directory", "Memory files", "Platform",
                  "Indexed chunks", "Stored images", "Database"):
        assert label in output
