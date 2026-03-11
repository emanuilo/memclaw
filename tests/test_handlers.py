"""Tests for Telegram handlers — double storage prevention (spec #7)."""
from __future__ import annotations

import inspect
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ────────────────────────────────────────────────────────────────────
# Spec #7: Double Storage Prevention
# ────────────────────────────────────────────────────────────────────

class TestHandlerSourceCode:
    """Static analysis tests — verify that handlers don't call store.save()
    for voice transcriptions and link summaries."""

    def test_handle_voice_no_store_save(self):
        """handle_voice should NOT call store.save() for transcriptions."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_voice)
        assert "store.save" not in source, (
            "handle_voice still calls store.save — voice transcriptions "
            "should be left for the agent to decide"
        )

    def test_handle_voice_has_not_saved_directive(self):
        """handle_voice should include 'NOT been saved yet' in the prompt."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_voice)
        assert "NOT been saved yet" in source

    def test_handle_text_no_store_save(self):
        """handle_text should NOT call store.save() for link summaries."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_text)
        assert "store.save" not in source

    def test_handle_text_has_not_saved_directive(self):
        """handle_text should include 'NOT been saved yet' for links."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_text)
        assert "NOT been saved yet" in source

    def test_handle_photo_no_store_save(self):
        """handle_photo should NOT call store.save() for link summaries."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_photo)
        assert "store.save" not in source

    def test_handle_photo_has_not_saved_directive(self):
        """handle_photo should include 'NOT been saved yet' for links."""
        from memclaw.bot.handlers import MessageHandlers
        source = inspect.getsource(MessageHandlers.handle_photo)
        assert "NOT been saved yet" in source


class TestAgentsFile:
    """Verify AGENTS.md (the externalized system prompt) has the right content."""

    def _read_agents(self) -> str:
        from pathlib import Path
        # Read the default AGENTS.md that ships with the project
        agents_path = Path.home() / ".memclaw" / "AGENTS.md"
        assert agents_path.exists(), "AGENTS.md not found at ~/.memclaw/AGENTS.md"
        return agents_path.read_text()

    def test_mentions_permanent_memory(self):
        content = self._read_agents()
        assert "permanent" in content.lower()
        assert "memory_save" in content

    def test_mentions_not_saved_yet(self):
        content = self._read_agents()
        assert "NOT" in content
        assert "saved" in content.lower()

    def test_mentions_voice_not_saved(self):
        content = self._read_agents()
        assert "Voice message" in content

    def test_mentions_link_not_saved(self):
        content = self._read_agents()
        assert "Link summary" in content

    def test_has_user_instructions_section(self):
        content = self._read_agents()
        assert "User instructions" in content
