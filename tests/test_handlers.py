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

        update.message.reply_text.assert_called_once_with("Hello!", parse_mode="Markdown")


class TestAgentsFile:
    """Verify AGENTS.md (the externalized system prompt) has the right content."""

    def _read_agents(self, tmp_config) -> str:
        agents_path = tmp_config.agent_file
        assert agents_path.exists(), f"AGENTS.md not found at {agents_path}"
        return agents_path.read_text()

    def test_mentions_permanent_memory(self, tmp_config):
        content = self._read_agents(tmp_config)
        assert "permanent" in content.lower()
        assert "memory_save" in content

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


# ────────────────────────────────────────────────────────────────────
# WhatsApp self-chat scoping (regression: outgoing DMs to friends
# were being processed because IsFromMe alone matches them too)
# ────────────────────────────────────────────────────────────────────

class TestWhatsAppSelfChatOnly:
    """Read whatsapp_handlers.py source directly to avoid needing neonize installed."""

    def _source(self) -> str:
        from pathlib import Path
        path = Path(__file__).parent.parent / "memclaw" / "bot" / "whatsapp_handlers.py"
        return path.read_text()

    def test_check_sender_requires_self_chat(self):
        """_check_sender must compare Chat.User to Sender.User, not just IsFromMe."""
        import ast
        src = self._source()
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_check_sender"
        )
        body_src = ast.unparse(fn)
        assert "Chat.User" in body_src and "Sender.User" in body_src, (
            "_check_sender must require Chat.User == Sender.User to scope to the "
            "self-chat — IsFromMe alone matches outgoing DMs to friends too"
        )

    def test_check_sender_behaviour(self):
        """Simulate the three relevant cases against the real method."""
        from types import SimpleNamespace
        import ast

        src = self._source()
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_check_sender"
        )
        ns: dict = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<_check_sender>", "exec"), ns)
        check = ns["_check_sender"]

        def ev(*, is_group: bool, is_from_me: bool, chat_user: str, sender_user: str):
            return SimpleNamespace(
                Info=SimpleNamespace(MessageSource=SimpleNamespace(
                    IsGroup=is_group,
                    IsFromMe=is_from_me,
                    Chat=SimpleNamespace(User=chat_user),
                    Sender=SimpleNamespace(User=sender_user),
                ))
            )

        self_note = ev(is_group=False, is_from_me=True, chat_user="me", sender_user="me")
        out_to_friend = ev(is_group=False, is_from_me=True, chat_user="friend", sender_user="me")
        in_from_friend = ev(is_group=False, is_from_me=False, chat_user="friend", sender_user="friend")
        group = ev(is_group=True, is_from_me=True, chat_user="grp", sender_user="me")

        assert check(None, self_note) is True
        assert check(None, out_to_friend) is False, "outgoing DM to friend must be ignored"
        assert check(None, in_from_friend) is False
        assert check(None, group) is False
