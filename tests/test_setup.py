"""Tests for the setup wizard (`run_setup`), driven by the settings in memclaw.config."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from memclaw import setup
from memclaw.config import MemclawConfig, all_settings


def _envs(settings) -> list[str]:
    return [s.env for s in settings]


class TestQuestions:
    def test_asked_on_every_install(self):
        assert _envs(setup.questions()) == ["OPENAI_API_KEY"]

    def test_asked_per_platform(self):
        assert _envs(setup.questions("telegram")) == ["TELEGRAM_BOT_TOKEN", "ALLOWED_USER_IDS"]
        assert _envs(setup.questions("slack")) == [
            "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_CHANNELS", "SLACK_ALLOWED_USERS",
        ]
        assert setup.questions("whatsapp") == setup.questions("terminal") == []

    def test_every_question_belongs_to_a_known_platform_or_none(self):
        platforms = {name for name, _ in setup.PLATFORMS}
        asked = setup.questions() + [s for p in platforms for s in setup.questions(p)]
        assert sorted(_envs(asked)) == sorted(s.env for s in all_settings() if s.prompt)


@pytest.fixture
def wizard(tmp_path: Path, monkeypatch):
    """Run `run_setup` with every prompt mocked and ~/.memclaw/.env in tmp_path.

    Returns a function(answers, platform, existing="") -> (env file text, prompts).
    """
    env_file = tmp_path / ".env"
    monkeypatch.setattr("memclaw.config.ENV_FILE", env_file)
    for s in all_settings():
        monkeypatch.delenv(s.env, raising=False)
    config = MemclawConfig(memory_dir=tmp_path / "m")

    def _run(answers: list[str], platform: str, existing: str = ""):
        if existing:
            env_file.write_text(existing)
        prompts: list[str] = []

        def _answer(_console, text):
            prompts.append(text)
            return answers[len(prompts) - 1]

        backend = ({"CLAUDE_CODE_OAUTH_TOKEN": "oat-token"}, ["ANTHROPIC_API_KEY"])
        with patch.object(setup, "_select_backend", return_value="claude"), \
                patch("memclaw.backends.ClaudeAgentBackend.wizard_setup", return_value=backend), \
                patch.object(setup, "_select_platform", return_value=platform), \
                patch.object(setup, "masked_input", side_effect=_answer), \
                patch.object(setup, "_check_openai"):
            setup.run_setup(config)
        return env_file.read_text(), prompts

    with patch.dict(os.environ):  # run_setup syncs its answers into os.environ
        yield _run


class TestRunSetup:
    def test_first_run_writes_the_answers(self, wizard):
        text, prompts = wizard(["sk-openai-1234", "xoxb-1", "xapp-1", "", "U1"], "slack")
        assert prompts == [
            "OpenAI API key (for embeddings + voice transcription) (required)",
            "Slack bot token (xoxb-...) (required)",
            "Slack app-level token for Socket Mode (xapp-...) (required)",
            "Allowed Slack channel IDs (comma-separated) (optional)",
            "Allowed Slack user IDs (comma-separated) (optional)",
        ]
        assert text == (
            "AGENT_BACKEND=claude\nCLAUDE_CODE_OAUTH_TOKEN=oat-token\n"
            "OPENAI_API_KEY=sk-openai-1234\nMEMCLAW_PLATFORM=slack\n"
            "SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nSLACK_ALLOWED_USERS=U1\n"
        )
        assert os.environ["SLACK_BOT_TOKEN"] == "xoxb-1"  # synced for the config made next

    def test_enter_keeps_saved_values_and_other_platforms_keys(self, wizard):
        existing = ("ANTHROPIC_API_KEY=sk-ant-old\nOPENAI_API_KEY=sk-openai-1234\n"
                    "TELEGRAM_BOT_TOKEN=123:abc\nALLOWED_USER_IDS=42\n")
        text, prompts = wizard([""], "whatsapp", existing)
        assert prompts == ["OpenAI API key (for embeddings + voice transcription) [sk-o...1234]"]
        assert text == (
            "OPENAI_API_KEY=sk-openai-1234\nTELEGRAM_BOT_TOKEN=123:abc\nALLOWED_USER_IDS=42\n"
            "AGENT_BACKEND=claude\nCLAUDE_CODE_OAUTH_TOKEN=oat-token\nMEMCLAW_PLATFORM=whatsapp\n"
        )
        assert "ANTHROPIC_API_KEY" not in os.environ  # dropped by the backend

    def test_a_missing_required_answer_stops_before_saving(self, wizard, tmp_path: Path):
        with pytest.raises(SystemExit):
            wizard(["sk-openai-1234", "", ""], "telegram")
        assert not (tmp_path / ".env").exists()
