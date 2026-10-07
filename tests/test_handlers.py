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


# ────────────────────────────────────────────────────────────────────
# /new, /model, /effort
# ────────────────────────────────────────────────────────────────────

def _telegram_handlers(*, allowed: bool = True):
    from memclaw.bot.handlers import MessageHandlers

    with patch.object(MessageHandlers, "__init__", lambda self, *a, **kw: None):
        handlers = MessageHandlers.__new__(MessageHandlers)
    handlers.agent = MagicMock()
    handlers.agent.reset_conversation = AsyncMock()
    handlers.config = MagicMock()
    handlers.config.allowed_user_ids_list = [1] if allowed else []
    return handlers


def _command_update(chat_id: int = 99):
    update = MagicMock()
    update.effective_user.id = 1
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    return update


class _SelectableBackend:
    """Minimal backend exposing the model-selection surface."""

    display_name = "Fake"
    supports_model_selection = True

    def __init__(self):
        from memclaw.backends.claude_models import ModelInfo

        self.models = [
            ModelInfo("claude-opus-5", "Claude Opus 5", "", ["low", "medium", "high"]),
            ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5", "", []),
        ]
        self.model = "claude-opus-5"
        self.effort = "high"

    async def list_models(self):
        return self.models

    async def effort_levels(self):
        info = next((m for m in self.models if m.id == self.model), None)
        return info.effort_levels if info else None

    async def set_model(self, model_id):
        if model_id not in [m.id for m in self.models]:
            raise ValueError(f"Unknown model '{model_id}'.")
        self.model = model_id
        if model_id == "claude-haiku-4-5":
            self.effort = None
            return "Claude Haiku 4.5 doesn't support an effort setting, so none is used."
        return ""

    async def set_effort(self, level):
        if level not in await self.effort_levels():
            raise ValueError(f"The current model doesn't support '{level}'.")
        self.effort = level


class TestNewCommand:
    @pytest.mark.asyncio
    async def test_resets_the_chat(self):
        handlers = _telegram_handlers()
        update = _command_update(chat_id=99)
        await handlers.new_command(update, MagicMock())
        handlers.agent.reset_conversation.assert_awaited_once_with("99")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ignores_unknown_users(self):
        handlers = _telegram_handlers(allowed=False)
        update = _command_update()
        await handlers.new_command(update, MagicMock())
        handlers.agent.reset_conversation.assert_not_awaited()
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slack_new_message_resets_the_channel(self):
        from memclaw.bot.slack_handlers import SlackHandlers

        with patch.object(SlackHandlers, "__init__", lambda self, *a, **kw: None):
            handlers = SlackHandlers.__new__(SlackHandlers)
        handlers.config = MagicMock()
        handlers.config.slack_allowed_channels_list = []
        handlers.config.slack_allowed_users_list = []
        handlers.agent = MagicMock()
        handlers.agent.reset_conversation = AsyncMock()
        handlers.agent.handle = AsyncMock()
        say = AsyncMock()

        await handlers._route_event(
            {"channel": "C1", "user": "U1", "text": "<@UBOT> /new", "ts": "1.0"},
            say, MagicMock(),
        )

        handlers.agent.reset_conversation.assert_awaited_once_with("C1")
        handlers.agent.handle.assert_not_awaited()
        say.assert_awaited_once()


class TestModelCommand:
    @pytest.mark.asyncio
    async def test_lists_models_numbered(self):
        from memclaw.bot.handlers import model_command_reply

        reply = await model_command_reply(_SelectableBackend(), [])
        assert "Model: claude-opus-5" in reply
        assert "Effort: high" in reply
        assert "1. Claude Opus 5 - claude-opus-5  (current)" in reply
        assert "2. Claude Haiku 4.5 - claude-haiku-4-5" in reply

    @pytest.mark.asyncio
    async def test_switch_by_number_reports_effort_change(self):
        from memclaw.bot.handlers import model_command_reply

        backend = _SelectableBackend()
        reply = await model_command_reply(backend, ["2"])
        assert backend.model == "claude-haiku-4-5"
        assert "Switched to claude-haiku-4-5." in reply
        assert "doesn't support an effort" in reply
        assert "conversation is kept" in reply

    @pytest.mark.asyncio
    async def test_switch_by_id(self):
        from memclaw.bot.handlers import model_command_reply

        backend = _SelectableBackend()
        backend.model = "claude-haiku-4-5"
        await model_command_reply(backend, ["claude-opus-5"])
        assert backend.model == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_bad_number_and_unknown_id(self):
        from memclaw.bot.handlers import model_command_reply

        backend = _SelectableBackend()
        assert "no model number 9" in await model_command_reply(backend, ["9"])
        assert "Unknown model" in await model_command_reply(backend, ["claude-nope"])
        assert backend.model == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_unsupported_backend(self):
        from memclaw.bot.handlers import effort_command_reply, model_command_reply

        backend = MagicMock(supports_model_selection=False, display_name="Cursor SDK")
        assert "isn't supported for the Cursor SDK backend" in await model_command_reply(backend, [])
        assert "isn't supported for the Cursor SDK backend" in await effort_command_reply(backend, ["low"])

    @pytest.mark.asyncio
    async def test_telegram_handler_passes_args(self):
        handlers = _telegram_handlers()
        handlers.agent.backend = _SelectableBackend()
        update = _command_update()
        context = MagicMock()
        context.args = ["2"]
        await handlers.model_command(update, context)
        assert handlers.agent.backend.model == "claude-haiku-4-5"
        update.message.reply_text.assert_awaited_once()


class TestEffortCommand:
    @pytest.mark.asyncio
    async def test_shows_supported_levels(self):
        from memclaw.bot.handlers import effort_command_reply

        reply = await effort_command_reply(_SelectableBackend(), [])
        assert "Effort: high" in reply
        assert "low, medium, high" in reply

    @pytest.mark.asyncio
    async def test_model_without_effort(self):
        from memclaw.bot.handlers import effort_command_reply

        backend = _SelectableBackend()
        backend.model, backend.effort = "claude-haiku-4-5", None
        reply = await effort_command_reply(backend, [])
        assert "doesn't support an effort setting" in reply

    @pytest.mark.asyncio
    async def test_sets_level(self):
        from memclaw.bot.handlers import effort_command_reply

        backend = _SelectableBackend()
        reply = await effort_command_reply(backend, ["low"])
        assert backend.effort == "low"
        assert "Effort set to low." in reply

    @pytest.mark.asyncio
    async def test_rejected_level(self):
        from memclaw.bot.handlers import effort_command_reply

        backend = _SelectableBackend()
        reply = await effort_command_reply(backend, ["max"])
        assert "doesn't support 'max'" in reply
        assert backend.effort == "high"
