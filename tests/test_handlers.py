"""Tests for Telegram handlers — double storage prevention (spec #7) + typing indicator."""
from __future__ import annotations

import asyncio
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


# ────────────────────────────────────────────────────────────────────
# Typing indicator
# ────────────────────────────────────────────────────────────────────

class TestTypingIndicator:
    def _make_handlers(self):
        from memclaw.bot.handlers import MessageHandlers

        with patch.object(MessageHandlers, "__init__", lambda self, *a, **kw: None):
            handlers = MessageHandlers.__new__(MessageHandlers)
        handlers.agent = MagicMock()
        handlers.config = MagicMock()
        handlers.openai_client = MagicMock()
        handlers.link_processor = MagicMock()
        return handlers

    @pytest.mark.asyncio
    async def test_typing_sent_during_processing(self):
        """_send_with_typing should send ChatAction.TYPING while agent runs."""
        from telegram.constants import ChatAction

        handlers = self._make_handlers()
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_photo = AsyncMock()

        async def slow_handle(prompt, **kw):
            await asyncio.sleep(0.05)
            return ("Response", [])

        handlers.agent.handle = slow_handle

        await handlers._send_with_typing(update, context, "Hi")

        context.bot.send_chat_action.assert_called()
        call_args = context.bot.send_chat_action.call_args
        assert call_args.kwargs.get("action") == ChatAction.TYPING or \
               (call_args.args and ChatAction.TYPING in call_args.args)

    @pytest.mark.asyncio
    async def test_response_sent_after_agent(self):
        """_send_with_typing should send the agent response via reply_text."""
        handlers = self._make_handlers()
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_photo = AsyncMock()

        handlers.agent.handle = AsyncMock(return_value=("Hello!", []))

        await handlers._send_with_typing(update, context, "Hi")

        update.message.reply_text.assert_called_once_with("Hello!")


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
