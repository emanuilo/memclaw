"""Tests for the system prompt a session starts with, and its fingerprint."""
from __future__ import annotations

from pathlib import Path

import pytest

from memclaw.config import ClaudeSettings, MemclawConfig, OpenAISettings
from memclaw.instructions import (
    _MEMORY_PROMPT_CHARS,
    _SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
    instructions_fingerprint,
    load_agent_instructions,
    load_permanent_memory,
)


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai=OpenAISettings(api_key="test-openai-key"),
        claude=ClaudeSettings(api_key="test-anthropic-key"),
    )


class TestPermanentMemoryPrompt:
    def test_small_memory_included_in_full(self, cfg: MemclawConfig):
        cfg.memory_file.write_text("## Key Facts\n\n- I like Python")
        assert load_permanent_memory(cfg) == "## Key Facts\n\n- I like Python"

    def test_runaway_memory_is_truncated(self, cfg: MemclawConfig):
        cfg.memory_file.write_text("x" * (_MEMORY_PROMPT_CHARS + 100))
        text = load_permanent_memory(cfg)
        assert len(text) < _MEMORY_PROMPT_CHARS + 100
        assert "truncated" in text

    def test_empty_memory(self, cfg: MemclawConfig):
        cfg.memory_file.write_text("")
        assert load_permanent_memory(cfg) == "(empty)"


class TestAgentInstructions:
    def test_agents_md_is_used(self, cfg: MemclawConfig):
        cfg.agent_file.write_text("  You are Test.\n")
        assert load_agent_instructions(cfg) == "You are Test."

    def test_default_without_agents_md(self, cfg: MemclawConfig):
        cfg.agent_file.unlink(missing_ok=True)
        assert load_agent_instructions(cfg) == "You are Memclaw, a personal memory assistant."


class TestSystemPrompt:
    def test_holds_instructions_then_memory(self):
        prompt = build_system_prompt("You are Test.", "- likes tea")
        assert prompt.startswith("You are Test.\n\n=== REPLY FORMATTING ===")
        assert prompt.index("=== PERMANENT MEMORY (MEMORY.md) ===") < prompt.index("- likes tea")

    def test_explains_references(self):
        assert "file_read" in _SYSTEM_PROMPT_TEMPLATE and "→ memory/" in _SYSTEM_PROMPT_TEMPLATE

    def test_fingerprint_is_per_backend_and_stable(self):
        assert instructions_fingerprint("claude") == instructions_fingerprint("claude")
        assert instructions_fingerprint("claude") != instructions_fingerprint("cursor")
        assert len(instructions_fingerprint("claude")) == 16
