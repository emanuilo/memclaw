"""Tests for MemclawConfig: the settings, their loader, and the files that
must list every setting (.env.example, docs)."""
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from memclaw.config import (
    ClaudeSettings,
    CursorSettings,
    MemclawConfig,
    MemorySettings,
    OpenAISettings,
    SlackSettings,
    TelegramSettings,
    all_settings,
)

ROOT = Path(__file__).resolve().parent.parent
ENV_NAMES = [s.env for s in all_settings()]


@pytest.fixture
def clean_env(monkeypatch):
    """No setting in the environment (importing memclaw.config loads the
    developer's ~/.memclaw/.env into it)."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _cfg(tmp_path: Path, **kwargs) -> MemclawConfig:
    return MemclawConfig(memory_dir=tmp_path / "m", **kwargs)


# ── Defaults and explicit values ────────────────────────────────────


def test_memory_defaults(tmp_path: Path):
    memory = _cfg(tmp_path).memory
    assert memory.consolidation_threshold == 1
    assert memory.decay_half_life_days == 30
    assert memory.mmr_lambda == 0.7


def test_custom_values(tmp_path: Path):
    cfg = _cfg(tmp_path, memory=MemorySettings(
        consolidation_threshold=3, decay_half_life_days=60, mmr_lambda=0.5))
    assert cfg.memory.consolidation_threshold == 3
    assert cfg.memory.decay_half_life_days == 60
    assert cfg.memory.mmr_lambda == 0.5


def test_unset_settings_get_their_defaults(tmp_path: Path, clean_env):
    cfg = _cfg(tmp_path)
    assert (cfg.agent_backend, cfg.platform) == ("", "")
    assert cfg.openai == OpenAISettings(api_key="")
    assert cfg.claude == ClaudeSettings()
    assert cfg.claude.model == cfg.claude.effort == ""  # empty = the backend's built-in default
    assert cfg.cursor.mcp_http_port == 17373
    assert cfg.telegram.allowed_user_ids_list == []
    assert cfg.slack.allowed_channels_list == cfg.slack.allowed_users_list == []


def test_every_setting_is_read_from_its_variable(tmp_path: Path, monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.setenv(name, "1234" if name == "MEMCLAW_MCP_PORT" else f"v-{name}")
    cfg = _cfg(tmp_path)
    assert cfg.platform == "v-MEMCLAW_PLATFORM"
    assert cfg.openai.api_key == "v-OPENAI_API_KEY"
    assert cfg.claude.oauth_token == "v-CLAUDE_CODE_OAUTH_TOKEN"
    assert cfg.claude.consolidation_effort == "v-CLAUDE_CONSOLIDATION_EFFORT"
    assert cfg.cursor.mcp_http_port == 1234
    assert cfg.telegram.allowed_user_ids == "v-ALLOWED_USER_IDS"
    assert cfg.slack.allowed_users == "v-SLACK_ALLOWED_USERS"


def test_explicit_values_win_over_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "from-env")
    monkeypatch.setenv("CLAUDE_EFFORT", "low")
    cfg = _cfg(tmp_path, claude=ClaudeSettings(model="claude-sonnet-5", effort="high"))
    assert cfg.claude.model == "claude-sonnet-5"
    assert cfg.claude.effort == "high"


def test_an_empty_explicit_value_falls_back_to_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert _cfg(tmp_path, openai=OpenAISettings(api_key="")).openai.api_key == "from-env"


def test_a_group_made_on_its_own_still_reads_its_other_fields(monkeypatch):
    monkeypatch.setenv("CURSOR_MODEL", "composer-3")
    settings = CursorSettings(api_key="k")
    assert (settings.api_key, settings.model) == ("k", "composer-3")


def test_claude_cli_model_vars_are_not_picked_up(tmp_path: Path, monkeypatch, isolate_claude_env):
    """ANTHROPIC_MODEL belongs to the Claude CLI; exporting it for Claude Code
    must not change which model Memclaw runs."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")
    assert _cfg(tmp_path).claude.model == ""


def test_rolling_history_settings_are_gone(tmp_path: Path):
    """Conversations live in the backend session now; there is no window."""
    cfg = _cfg(tmp_path)
    assert not hasattr(cfg, "conversation_history_limit")
    assert not hasattr(cfg, "conversation_history_window_minutes")


def test_secrets_stay_out_of_repr(tmp_path: Path):
    cfg = _cfg(tmp_path, openai=OpenAISettings(api_key="sk-hidden"),
               slack=SlackSettings(bot_token="xoxb-hidden", allowed_users="U1"))
    text = repr(cfg)
    assert "hidden" not in text
    assert "allowed_users='U1'" in text


# ── Parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "port"), [("8080", 8080), (" 8080 ", 8080), ("", 17373), ("  ", 17373)])
def test_mcp_port_is_parsed(tmp_path: Path, monkeypatch, raw: str, port: int):
    monkeypatch.setenv("MEMCLAW_MCP_PORT", raw)
    assert _cfg(tmp_path).cursor.mcp_http_port == port


@pytest.mark.parametrize(("raw", "message"), [
    ("abc", "MEMCLAW_MCP_PORT must be an integer, got 'abc'"),
    ("0", "MEMCLAW_MCP_PORT must be between 1 and 65535, got 0"),
    ("65536", "MEMCLAW_MCP_PORT must be between 1 and 65535, got 65536"),
])
def test_bad_mcp_port_is_refused(tmp_path: Path, monkeypatch, raw: str, message: str):
    monkeypatch.setenv("MEMCLAW_MCP_PORT", raw)
    with pytest.raises(ValueError, match=re.escape(message)):
        _cfg(tmp_path)


def test_comma_separated_lists():
    assert TelegramSettings(allowed_user_ids=" 1, 2,,3 ").allowed_user_ids_list == [1, 2, 3]
    slack = SlackSettings(allowed_channels="C1, C2,", allowed_users=" U1 ")
    assert slack.allowed_channels_list == ["C1", "C2"]
    assert slack.allowed_users_list == ["U1"]


# ── Where values come from ──────────────────────────────────────────


def _load_in_fresh_process(home: Path, env: dict[str, str], names: list[str]) -> list[str]:
    """Import memclaw.config in a new process (it loads ~/.memclaw/.env on
    import) with HOME=*home* and *env*, and return the values of *names*."""
    code = (
        "import sys; from memclaw.config import ClaudeSettings as C; c = C(); "
        "print('|'.join(getattr(c, n) for n in sys.argv[1:]))"
    )
    child_env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    child_env.update(env, HOME=str(home))
    out = subprocess.run([sys.executable, "-c", code, *names], env=child_env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().split("|")


def test_environment_wins_over_the_saved_env_file(tmp_path: Path):
    (tmp_path / ".memclaw").mkdir()
    (tmp_path / ".memclaw" / ".env").write_text("CLAUDE_MODEL=from-file\nCLAUDE_EFFORT=high\n")
    values = _load_in_fresh_process(tmp_path, {"CLAUDE_MODEL": "from-env"}, ["model", "effort"])
    assert values == ["from-env", "high"]


def test_reload_env_file_lets_the_saved_file_win(tmp_path: Path, monkeypatch):
    from memclaw import config

    env_file = tmp_path / ".env"
    env_file.write_text("CLAUDE_MODEL=just-saved\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("CLAUDE_MODEL", "stale")
    config.reload_env_file()
    assert ClaudeSettings().model == "just-saved"


def test_set_environment_sets_and_drops(monkeypatch):
    from memclaw import config

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "old")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config.set_environment({"ANTHROPIC_API_KEY": "sk", "CLAUDE_MODEL": ""},
                           drop_keys=["CLAUDE_CODE_OAUTH_TOKEN"])
    assert ClaudeSettings().api_key == "sk"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ


def test_update_env_file_merges_and_drops(tmp_path: Path, monkeypatch):
    from memclaw import config

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=k\nCLAUDE_MODEL=old\nCLAUDE_EFFORT=high\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)

    config.update_env_file({"CLAUDE_MODEL": "claude-haiku-4-5"}, drop_keys=["CLAUDE_EFFORT"])

    assert env_file.read_text() == "OPENAI_API_KEY=k\nCLAUDE_MODEL=claude-haiku-4-5\n"


# ── Every setting is listed where users look for it ─────────────────


def test_all_settings_are_unique_and_grouped():
    assert len(set(ENV_NAMES)) == len(ENV_NAMES)
    groups = {s.group for s in all_settings()}
    assert groups == {"general", "openai", "claude", "cursor", "telegram", "slack"}


def test_env_example_lists_exactly_the_settings():
    text = (ROOT / ".env.example").read_text()
    listed = re.findall(r"^#? ?([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE)
    for name in ENV_NAMES:
        assert name in listed, (
            f"{name} is read by memclaw.config but missing from .env.example: "
            f"add `{name}=...` (or `# {name}=...` if optional)")
    for name in listed:
        assert name in ENV_NAMES, (
            f"{name} is in .env.example but memclaw.config doesn't read it")


def test_configuration_docs_list_exactly_the_settings():
    text = (ROOT / "docs" / "configuration.md").read_text()
    listed = re.findall(r"^\| `([A-Z][A-Z0-9_]+)` \|", text, flags=re.MULTILINE)
    for name in ENV_NAMES:
        assert name in listed, (
            f"{name} is read by memclaw.config but missing from the table in "
            f"docs/configuration.md")
    for name in listed:
        assert name in ENV_NAMES, (
            f"{name} is in docs/configuration.md but memclaw.config doesn't read it")


def test_memory_docs_list_every_memory_setting():
    text = (ROOT / "docs" / "memory-architecture.md").read_text()
    for f in fields(MemorySettings):
        assert f"`memory.{f.name}`" in text, (
            f"MemorySettings.{f.name} is missing from the Configuration Reference "
            f"in docs/memory-architecture.md")
