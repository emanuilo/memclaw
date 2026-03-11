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


class TestSystemPrompt:
    """Verify the system prompt includes guidance for unsaved content."""

    def test_prompt_mentions_permanent_memory(self):
        from memclaw.agent import SYSTEM_PROMPT
        assert "permanent" in SYSTEM_PROMPT.lower()
        assert "memory_save" in SYSTEM_PROMPT

    def test_prompt_mentions_not_saved_yet(self):
        from memclaw.agent import SYSTEM_PROMPT
        assert "NOT" in SYSTEM_PROMPT
        assert "saved" in SYSTEM_PROMPT

    def test_prompt_mentions_voice_not_saved(self):
        from memclaw.agent import SYSTEM_PROMPT
        assert "Voice message" in SYSTEM_PROMPT
        assert "NOT yet saved" in SYSTEM_PROMPT or "NOT been saved" in SYSTEM_PROMPT

    def test_prompt_mentions_link_not_saved(self):
        from memclaw.agent import SYSTEM_PROMPT
        assert "Link summary" in SYSTEM_PROMPT
        assert "NOT yet saved" in SYSTEM_PROMPT or "NOT been saved" in SYSTEM_PROMPT
