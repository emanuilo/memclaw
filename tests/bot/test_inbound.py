"""The shared message pipeline (memclaw.bot.inbound). Its prompts and
replies are checked on every platform in test_parity.py; here: what the
pipeline must not do, and the agent instructions its prompts rely on."""
from __future__ import annotations

import inspect

import memclaw.bot.inbound as inbound
from memclaw.bot.inbound import is_allowed


def test_pipeline_never_saves_on_its_own():
    """Voice transcriptions, link summaries and photos are left for the agent
    to save (spec #7: no double storage): the prompts say they are NOT saved
    yet (test_parity.py) and nothing here calls the store."""
    assert "store.save" not in inspect.getsource(inbound)


def test_is_allowed():
    assert is_allowed("a", ["a", "b"])
    assert not is_allowed("c", ["a", "b"])
    assert not is_allowed("c", [])  # an empty allowlist lets nobody in


class TestAgentsFile:
    """Verify AGENTS.md (the externalized system prompt) has the right content."""

    def _read_agents(self, tmp_config) -> str:
        agents_path = tmp_config.agent_file
        assert agents_path.exists(), f"AGENTS.md not found at {agents_path}"
        return agents_path.read_text()

    def test_mentions_permanent_memory(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "MEMORY.md" in content
        assert "memory_save" in content
        assert "permanent=true" not in content

    def test_mentions_not_saved_yet(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "NOT" in content
        assert "saved" in content.lower()

    def test_mentions_voice_not_saved(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "Voice message" in content

    def test_mentions_link_not_saved(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "Link summary" in content

    def test_has_user_instructions_section(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "User instructions" in content
