"""Telegram: the typing indicator, the "Working…" status message and the
/new, /model and /effort commands. (What Telegram shares with the other
platforms is tested in test_parity.py.)"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ────────────────────────────────────────────────────────────────────
# Typing indicator
# ────────────────────────────────────────────────────────────────────

class TestTypingIndicator:
    def _make_handlers(self):
        return _telegram_handlers()

    @pytest.mark.asyncio
    async def test_typing_sent_during_processing(self):
        """_send_with_typing should send ChatAction.TYPING while agent runs."""
        from telegram.constants import ChatAction

        handlers = self._make_handlers()
        update = _command_update(chat_id=123)
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_photo = AsyncMock()

        async def slow_handle(prompt, **kw):
            await asyncio.sleep(0.05)
            return ("Response", [])

        handlers.inbound.agent.handle = slow_handle

        await handlers._send_with_typing(update.message, context, "Hi")

        context.bot.send_chat_action.assert_called()
        call_args = context.bot.send_chat_action.call_args
        assert call_args.kwargs.get("action") == ChatAction.TYPING or \
               (call_args.args and ChatAction.TYPING in call_args.args)

    @pytest.mark.asyncio
    async def test_response_sent_after_agent(self):
        """_send_with_typing should send the agent response via reply_text."""
        handlers = self._make_handlers()
        update = _command_update(chat_id=123)
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_photo = AsyncMock()

        handlers.inbound.agent.handle = AsyncMock(return_value=("Hello!", []))

        await handlers._send_with_typing(update.message, context, "Hi")

        update.message.reply_text.assert_called_once_with("Hello!", parse_mode="Markdown")


# ────────────────────────────────────────────────────────────────────
# /new, /model, /effort
# ────────────────────────────────────────────────────────────────────

def _telegram_handlers(*, allowed: bool = True):
    """A TelegramBot whose agent is a mock (no backend, no vault)."""
    from memclaw.bot.inbound import Inbound
    from memclaw.bot.telegram import TelegramBot

    handlers = TelegramBot.__new__(TelegramBot)
    handlers.inbound = Inbound.__new__(Inbound)
    handlers.inbound.agent = MagicMock()
    handlers.inbound.agent.reset_conversation = AsyncMock()
    handlers.config = MagicMock()
    handlers.config.telegram.allowed_user_ids_list = [1] if allowed else []
    return handlers


def _command_update(chat_id: int = 99):
    update = MagicMock()
    update.effective_user.id = 1
    update.message.chat_id = chat_id
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
        handlers.inbound.agent.reset_conversation.assert_awaited_once_with("99")
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ignores_unknown_users(self):
        handlers = _telegram_handlers(allowed=False)
        update = _command_update()
        await handlers.new_command(update, MagicMock())
        handlers.inbound.agent.reset_conversation.assert_not_awaited()
        update.message.reply_text.assert_not_awaited()

def _buttons(keyboard):
    return [(b.text, b.callback_data) for row in keyboard.inline_keyboard for b in row]


def _callback_update(data: str, *, allowed_user: int = 1):
    update = MagicMock()
    update.effective_user.id = allowed_user
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


class TestModelCommand:
    @pytest.mark.asyncio
    async def test_menu_has_a_button_per_model(self):
        from memclaw.bot.telegram import model_menu

        text, keyboard = await model_menu(_SelectableBackend())
        assert "Model: Claude Opus 5 (claude-opus-5)" in text
        assert "Effort: high" in text
        assert _buttons(keyboard) == [
            ("✓ Claude Opus 5", "m:claude-opus-5"),
            ("Claude Haiku 4.5", "m:claude-haiku-4-5"),
        ]

    def test_long_model_id_falls_back_to_a_hash(self):
        from memclaw.bot.telegram import _model_key, model_callback_data

        assert model_callback_data("x" * 70) == f"mi:{_model_key('x' * 70)}"
        assert len(model_callback_data("x" * 70).encode()) <= 64
        assert model_callback_data("claude-opus-5") == "m:claude-opus-5"

    @pytest.mark.asyncio
    async def test_button_press_switches_and_edits_the_menu(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _callback_update("m:claude-haiku-4-5")

        await handlers.handle_callback(update, MagicMock())

        assert backend.model == "claude-haiku-4-5"
        update.callback_query.answer.assert_awaited_once()
        text = update.callback_query.edit_message_text.await_args.args[0]
        assert "Switched to claude-haiku-4-5." in text
        assert "doesn't support an effort" in text
        assert "conversation is kept" in text
        keyboard = update.callback_query.edit_message_text.await_args.kwargs["reply_markup"]
        assert ("✓ Claude Haiku 4.5", "m:claude-haiku-4-5") in _buttons(keyboard)

    @pytest.mark.asyncio
    async def test_index_button_press(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        from memclaw.bot.telegram import _model_key

        await handlers.handle_callback(
            _callback_update(f"mi:{_model_key('claude-haiku-4-5')}"), MagicMock(),
        )
        assert backend.model == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_stale_button_alerts(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _callback_update("m:claude-gone")
        await handlers.handle_callback(update, MagicMock())
        assert backend.model == "claude-opus-5"
        assert update.callback_query.answer.await_args.kwargs["show_alert"] is True
        update.callback_query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_button_press_from_unknown_user_is_ignored(self):
        handlers = _telegram_handlers(allowed=False)
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _callback_update("m:claude-haiku-4-5")
        await handlers.handle_callback(update, MagicMock())
        assert backend.model == "claude-opus-5"
        update.callback_query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_command_with_number_or_id(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _command_update()
        context = MagicMock()

        context.args = ["2"]
        await handlers.model_command(update, context)
        assert backend.model == "claude-haiku-4-5"

        context.args = ["claude-opus-5"]
        await handlers.model_command(update, context)
        assert backend.model == "claude-opus-5"
        assert update.message.reply_text.await_args.kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_bad_number_and_unknown_id(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _command_update()
        context = MagicMock()
        for arg, expected in (("9", "no model number 9"), ("claude-nope", "Unknown model")):
            context.args = [arg]
            await handlers.model_command(update, context)
            assert expected in update.message.reply_text.await_args.args[0]
        assert backend.model == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_unsupported_backend(self):
        from memclaw.bot.telegram import effort_menu, model_menu

        backend = MagicMock(supports_model_selection=False, display_name="Cursor SDK")
        text, keyboard = await model_menu(backend)
        assert "isn't supported for the Cursor SDK backend" in text
        assert keyboard is None
        text, keyboard = await effort_menu(backend)
        assert "isn't supported for the Cursor SDK backend" in text
        assert keyboard is None


class TestEffortCommand:
    @pytest.mark.asyncio
    async def test_menu_has_a_button_per_level(self):
        from memclaw.bot.telegram import effort_menu

        text, keyboard = await effort_menu(_SelectableBackend())
        assert "Effort: high" in text
        assert _buttons(keyboard) == [
            ("low", "e:low"), ("medium", "e:medium"), ("✓ high", "e:high"),
        ]

    @pytest.mark.asyncio
    async def test_model_without_effort_has_no_buttons(self):
        from memclaw.bot.telegram import effort_menu

        backend = _SelectableBackend()
        backend.model, backend.effort = "claude-haiku-4-5", None
        text, keyboard = await effort_menu(backend)
        assert "doesn't support an effort setting" in text
        assert keyboard is None

    @pytest.mark.asyncio
    async def test_button_press_sets_level(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _callback_update("e:low")
        await handlers.handle_callback(update, MagicMock())
        assert backend.effort == "low"
        text = update.callback_query.edit_message_text.await_args.args[0]
        assert "Effort set to low." in text

    @pytest.mark.asyncio
    async def test_command_sets_level(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _command_update()
        context = MagicMock()
        context.args = ["low"]
        await handlers.effort_command(update, context)
        assert backend.effort == "low"

    @pytest.mark.asyncio
    async def test_rejected_level(self):
        handlers = _telegram_handlers()
        handlers.inbound.agent.backend = backend = _SelectableBackend()
        update = _command_update()
        context = MagicMock()
        context.args = ["max"]
        await handlers.effort_command(update, context)
        assert "doesn't support 'max'" in update.message.reply_text.await_args.args[0]
        assert backend.effort == "high"


class TestStepStatusMessage:
    @pytest.mark.asyncio
    async def test_status_message_is_sent_edited_and_deleted(self):
        from memclaw.backends.base import ToolStep
        from memclaw.bot import telegram as handlers_mod

        handlers = _telegram_handlers()
        handlers._send_response = AsyncMock()
        status = MagicMock(edit_text=AsyncMock(), delete=AsyncMock())
        context = MagicMock()
        context.bot.send_message = AsyncMock(return_value=status)
        context.bot.send_chat_action = AsyncMock()

        async def handle(prompt, *, on_tool, **kwargs):
            await on_tool(ToolStep(1, "memory_search", "Searching memories: dog"))
            await on_tool(ToolStep(2, "memory_save", "Saving to memory"))
            return "Bruno", []

        handlers.inbound.agent.handle = handle
        with patch.object(handlers_mod, "STATUS_EDIT_INTERVAL_S", 0):
            await handlers._send_with_typing(_command_update().message, context, "hi")

        context.bot.send_message.assert_awaited_once()
        assert context.bot.send_message.await_args.args[1] == (
            "Working… step 1: Searching memories: dog"
        )
        status.edit_text.assert_awaited_once_with("Working… step 2: Saving to memory")
        status.delete.assert_awaited_once()
        handlers._send_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_status_message_without_tool_calls(self):
        handlers = _telegram_handlers()
        handlers._send_response = AsyncMock()
        context = MagicMock()
        context.bot.send_message = AsyncMock()
        context.bot.send_chat_action = AsyncMock()
        handlers.inbound.agent.handle = AsyncMock(return_value=("hello", []))

        await handlers._send_with_typing(_command_update().message, context, "hi")
        context.bot.send_message.assert_not_awaited()


class TestDescribeToolCall:
    def test_labels_and_truncation(self):
        from memclaw.tools import describe_tool_call

        assert describe_tool_call("memory_search", {"query": "dog  name"}) == (
            "Searching memories: dog name"
        )
        assert describe_tool_call("reminder_list", {}) == "Checking reminders"
        assert describe_tool_call("unknown_tool", {"x": 1}) == "unknown_tool"
        long = describe_tool_call("file_read", {"file_path": "a" * 200})
        assert long.endswith("…") and len(long) <= len("Reading: ") + 80
