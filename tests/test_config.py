"""Tests for new MemclawConfig fields (specs #1, #2, #4, #5)."""
from pathlib import Path

from memclaw.config import MemclawConfig


def test_default_conversation_history_limit(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.conversation_history_limit == 10


def test_default_consolidation_threshold(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.consolidation_threshold == 7


def test_default_decay_half_life_days(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.decay_half_life_days == 30


def test_default_mmr_lambda(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.mmr_lambda == 0.7


def test_custom_values(tmp_path: Path):
    cfg = MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="k",
        anthropic_api_key="k",
        conversation_history_limit=5,
        consolidation_threshold=3,
        decay_half_life_days=60,
        mmr_lambda=0.5,
    )
    assert cfg.conversation_history_limit == 5
    assert cfg.consolidation_threshold == 3
    assert cfg.decay_half_life_days == 60
    assert cfg.mmr_lambda == 0.5


def test_claude_model_and_effort_default_to_empty(tmp_path: Path, isolate_claude_env):
    """Empty is the signal to fall back to the backend built-in default."""
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.claude_model == ""
    assert cfg.claude_effort == ""


def test_claude_model_and_effort_read_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.claude_model == "claude-opus-5"
    assert cfg.claude_effort == "xhigh"


def test_explicit_claude_values_win_over_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "from-env")
    monkeypatch.setenv("CLAUDE_EFFORT", "low")
    cfg = MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="k",
        anthropic_api_key="k",
        claude_model="claude-sonnet-5",
        claude_effort="high",
    )
    assert cfg.claude_model == "claude-sonnet-5"
    assert cfg.claude_effort == "high"


def test_claude_cli_model_vars_are_not_picked_up(
    tmp_path: Path, monkeypatch, isolate_claude_env,
):
    """ANTHROPIC_MODEL belongs to the Claude CLI; exporting it for Claude Code
    must not change which model Memclaw runs."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.claude_model == ""
