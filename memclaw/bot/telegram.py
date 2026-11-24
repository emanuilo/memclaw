"""Telegram: receive messages, download their media, send replies, and the
/new, /model and /effort commands with their button menus.

What happens to a message in between is shared with the other platforms
(memclaw.bot.inbound).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from typing import Awaitable, Callable

from loguru import logger
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, NetworkError, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..backends import AgentBackend
from ..backends.base import ToolStep
from ..config import MemclawConfig
from . import allowlist_label, console, log_to, mask_user_id, require_setting
from .commands import (
    load_models,
    resolve_model_arg,
    select_effort,
    select_model,
    status_lines,
    unsupported,
)
from .inbound import TRANSCRIPTION_FAILED, Inbound, is_allowed


async def _typing_loop(bot, chat_id: int):
    """Send 'typing...' action every 4 seconds until cancelled."""
    try:
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass  # best-effort
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


# Shown in Telegram's command menu (registered in build_application).
BOT_COMMANDS = [
    ("new", "Start a fresh conversation"),
    ("model", "Show or change the model"),
    ("effort", "Show or change the reasoning effort"),
]

# Minimum seconds between edits of the "Working…" status message, to stay
# well under Telegram's rate limits.
STATUS_EDIT_INTERVAL_S = 2.0

# Telegram limits callback_data to 64 bytes.
CALLBACK_DATA_MAX_BYTES = 64


def _model_key(model_id: str) -> str:
    return hashlib.sha256(model_id.encode()).hexdigest()[:16]


def model_callback_data(model_id: str) -> str:
    """``m:<id>``, or ``mi:<hash of the id>`` if the id would exceed
    Telegram's limit. A hash rather than a list position, so a button still
    picks its own model if the list has changed since the menu was shown."""
    data = f"m:{model_id}"
    return data if len(data.encode()) <= CALLBACK_DATA_MAX_BYTES else f"mi:{_model_key(model_id)}"


async def model_menu(
    backend: AgentBackend, note: str = "",
) -> tuple[str, InlineKeyboardMarkup | None]:
    """`/model` reply: current settings plus one button per model."""
    if not backend.supports_model_selection:
        return unsupported(backend, "Switching models"), None
    models = await load_models(backend)
    lines = status_lines(backend, models, await backend.effort_levels())
    if note:
        lines += ["", note]
    lines.append("")
    if not models:
        lines.append("Couldn't load the model list right now; try /model again later.")
        return "\n".join(lines), None
    lines.append("Tap a model to switch, or send /model <id>.")
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            ("✓ " if m.id == backend.model else "") + m.display_name,
            callback_data=model_callback_data(m.id),
        )]
        for m in models
    ])
    return "\n".join(lines), keyboard


async def effort_menu(
    backend: AgentBackend, note: str = "",
) -> tuple[str, InlineKeyboardMarkup | None]:
    """`/effort` reply: current settings plus a row of level buttons."""
    if not backend.supports_model_selection:
        return unsupported(backend, "Changing the effort"), None
    levels = await backend.effort_levels()
    lines = status_lines(backend, await load_models(backend), levels)
    if note:
        lines += ["", note]
    lines.append("")
    if levels == []:
        lines.append("This model doesn't support an effort setting.")
        return "\n".join(lines), None
    if levels is None:
        lines.append("Send /effort <level> to change it.")
        return "\n".join(lines), None
    lines.append("Tap a level, or send /effort <level>.")
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            ("✓ " if lv == backend.effort else "") + lv, callback_data=f"e:{lv}",
        )
        for lv in levels
    ]])
    return "\n".join(lines), keyboard


def _command_arg(context: ContextTypes.DEFAULT_TYPE) -> str:
    return " ".join(context.args or []).strip()


def _who(update: Update) -> str:
    """The sender's masked id, for the log."""
    user = update.effective_user
    return mask_user_id(user.id if user else "")


class TelegramBot:
    """Receives Telegram messages and commands, and sends the replies."""

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.inbound = Inbound(config, "telegram")

    def attach_bot(self, bot) -> None:
        """Give the handler a reference to the Telegram Bot for reminder delivery."""
        async def send(chat_id: str, text: str) -> None:
            await bot.send_message(chat_id=int(chat_id), text=text)

        self.inbound.deliver_reminders(send)

    def _allowed(self, update: Update) -> bool:
        """Only the users in ALLOWED_USER_IDS; an empty list lets nobody in."""
        user = update.effective_user
        return user is not None and is_allowed(user.id, self.config.telegram.allowed_user_ids_list)

    def _message(self, update: Update) -> Message | None:
        """The update's message, if it comes from an allowed user."""
        return update.message if self._allowed(update) else None

    async def _send_response(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        response_text: str,
        found_images: list[dict],
    ):
        """Send agent response: images first, then text."""
        for img in found_images:
            try:
                await context.bot.send_photo(
                    chat_id=message.chat_id,
                    photo=img["file_id"],
                    caption=img.get("caption") or None,
                )
            except Exception as e:
                logger.exception(f"Failed to send image {img.get('file_id')}: {e}")

        if response_text:
            try:
                await message.reply_text(response_text[:4096], parse_mode="Markdown")
            except Exception as e:
                # Fall back to plain text if the agent emitted something Telegram's
                # legacy Markdown parser rejects (unbalanced * or _, etc.).
                logger.warning(f"Markdown parse failed, sending plain: {e}")
                await message.reply_text(response_text[:4096])

    async def _send_with_typing(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        *,
        image: tuple[bytes, str] | None = None,
    ):
        """Run the agent with a typing indicator, then send the full response."""
        chat_id = message.chat_id
        typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id))
        # One silent status message, edited as the agent works and deleted
        # once the reply is ready.
        status: Message | None = None
        last_edit = 0.0

        async def on_tool(step: ToolStep) -> None:
            nonlocal status, last_edit
            now = time.monotonic()
            if now - last_edit < STATUS_EDIT_INTERVAL_S:
                return
            last_edit = now
            text = f"Working… step {step.index}: {step.summary}"
            try:
                if status is None:
                    status = await context.bot.send_message(
                        chat_id, text, disable_notification=True,
                    )
                else:
                    await status.edit_text(text)
            except TelegramError as exc:
                logger.debug("Status update failed: {exc}", exc=exc)

        try:
            response_text, found_images = await self.inbound.ask(
                prompt, chat_id=str(chat_id), image=image, on_tool=on_tool,
            )
        finally:
            typing_task.cancel()
            if status is not None:
                with contextlib.suppress(TelegramError):
                    await status.delete()
        await self._send_response(message, context, response_text, found_images)

    # ── Commands ─────────────────────────────────────────────────────

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return

        await message.reply_text(
            "Hi! I'm your personal memory assistant powered by Memclaw.\n\n"
            "Just send me anything — text, photos, or voice messages.\n\n"
            "I'll automatically decide whether to remember it, search your "
            "memories, or retrieve images. No commands needed, just talk to me.\n\n"
            "/new - start a fresh conversation (your memories are kept)\n"
            "/model - show or change the model\n"
            "/effort - show or change the reasoning effort"
        )

    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return
        await message.reply_text(await self.inbound.new_conversation(str(message.chat_id)))

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return
        backend, arg, note = self.inbound.agent.backend, _command_arg(context), ""
        if arg and backend.supports_model_selection:
            try:
                note = await select_model(backend, await resolve_model_arg(backend, arg))
            except ValueError as exc:
                note = f"{exc} Pick one below."
        text, keyboard = await model_menu(backend, note)
        await message.reply_text(text, reply_markup=keyboard)

    async def effort_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return
        backend, arg, note = self.inbound.agent.backend, _command_arg(context), ""
        if arg and backend.supports_model_selection:
            try:
                note = await select_effort(backend, arg)
            except ValueError as exc:
                note = str(exc)
        text, keyboard = await effort_menu(backend, note)
        await message.reply_text(text, reply_markup=keyboard)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Button presses on the /model and /effort menus."""
        query = update.callback_query
        if query is None:
            return
        if not self._allowed(update):
            await query.answer("You're not allowed to use this bot.")
            return
        backend = self.inbound.agent.backend
        kind, _, value = (query.data or "").partition(":")
        if kind not in ("m", "mi", "e") or not backend.supports_model_selection:
            await query.answer()
            return
        try:
            if kind == "e":
                note = await select_effort(backend, value)
                await query.answer(f"Effort: {backend.effort}")
                text, keyboard = await effort_menu(backend, note)
            else:
                model_id = value
                if kind == "mi":
                    models = await load_models(backend)
                    model_id = next((m.id for m in models if _model_key(m.id) == value), "")
                    if not model_id:
                        raise ValueError("That model is no longer available; send /model again.")
                note = await select_model(backend, model_id)
                await query.answer(f"Model: {backend.model}")
                text, keyboard = await model_menu(backend, note)
        except ValueError as exc:
            await query.answer(str(exc)[:200], show_alert=True)
            return
        try:
            await query.edit_message_text(text, reply_markup=keyboard)
        except BadRequest as exc:  # e.g. "message is not modified"
            logger.debug("Editing the menu after a button press failed: {exc}", exc=exc)

    # ── Messages: everything goes through the shared pipeline ────────

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return

        text = message.text or ""
        logger.info(f"Text from user {_who(update)}: {text[:100]}")

        replied = message.reply_to_message
        quoted = (replied.text or replied.caption or "") if replied is not None else ""
        prompt = await self.inbound.text_prompt(text, quoted=quoted)
        await self._send_with_typing(message, context, prompt)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None:
            return

        photo = message.photo[-1]
        caption = message.caption or ""

        logger.info(f"Photo from user {_who(update)}, caption={caption!r}")

        file = await context.bot.get_file(photo.file_id)
        photo_bytes = bytes(await file.download_as_bytearray())
        logger.debug(f"Downloaded photo: {len(photo_bytes)} bytes")

        prompt = await self.inbound.photo_prompt(photo.file_id, caption)
        await self._send_with_typing(message, context, prompt, image=(photo_bytes, "image/jpeg"))

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self._message(update)
        if message is None or message.voice is None:
            return

        logger.info(f"Voice from user {_who(update)}")

        file = await context.bot.get_file(message.voice.file_id)
        voice_bytes = bytes(await file.download_as_bytearray())

        text = await self.inbound.transcribe(voice_bytes, filename="voice.ogg", mime_type="audio/ogg")
        if text is None:
            await message.reply_text(TRANSCRIPTION_FAILED)
            return
        logger.info(f"Transcribed: {text}")

        await self._send_with_typing(message, context, await self.inbound.voice_prompt(text))

    async def aclose(self):
        await self.inbound.aclose()


Handler = Callable[[TelegramBot, Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


def _route(method: Handler):
    """A python-telegram-bot callback that calls *method* on the TelegramBot
    (created in post_init, kept in bot_data)."""
    async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await method(context.bot_data["handlers"], update, context)
    return callback


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, (NetworkError, TimedOut)):
        logger.warning(f"Network blip ({type(err).__name__}): {err} — polling will retry")
        return
    logger.exception("Unhandled error in Telegram handler", exc_info=err)


def build_application(config: MemclawConfig) -> Application:
    """Build the Telegram bot: TelegramBot, command handlers, error
    handler and command menu, wired into a python-telegram-bot Application.
    """
    async def post_init(application: Application) -> None:
        handlers = TelegramBot(config)
        application.bot_data["handlers"] = handlers

        handlers.attach_bot(application.bot)
        await handlers.inbound.start()

        try:
            await application.bot.set_my_commands(
                [BotCommand(name, description) for name, description in BOT_COMMANDS]
            )
        except TelegramError as exc:
            logger.warning(f"Could not register the bot command menu: {exc}")

        logger.info("Memclaw bot initialized")

    async def post_shutdown(application: Application) -> None:
        handlers = application.bot_data.get("handlers")
        if handlers:
            await handlers.aclose()
            logger.info("Memclaw bot shut down cleanly")

    app = (
        Application.builder()
        .token(config.telegram.bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_error_handler(_on_error)
    for command, method in (
        ("start", TelegramBot.start_command),
        ("new", TelegramBot.new_command),
        ("model", TelegramBot.model_command),
        ("effort", TelegramBot.effort_command),
    ):
        app.add_handler(CommandHandler(command, _route(method)))
    app.add_handler(CallbackQueryHandler(_route(TelegramBot.handle_callback)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _route(TelegramBot.handle_text)))
    app.add_handler(MessageHandler(filters.PHOTO, _route(TelegramBot.handle_photo)))
    app.add_handler(MessageHandler(filters.VOICE, _route(TelegramBot.handle_voice)))

    return app


def run(config: MemclawConfig) -> None:
    """Start the Telegram bot (long polling) and block until it stops."""
    require_setting(config.telegram.bot_token, "TELEGRAM_BOT_TOKEN")

    log_to(config.memory_dir / "bot.log")

    app = build_application(config)

    masked = [mask_user_id(u) for u in config.telegram.allowed_user_ids_list]
    users = allowlist_label(masked, "ALLOWED_USER_IDS")
    console.print(f"[green]Starting Memclaw Telegram bot...[/green]  (allowed users: {users})")
    app.run_polling(allowed_updates=["message", "callback_query"])
