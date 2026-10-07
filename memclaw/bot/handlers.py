"""Telegram bot message handlers for Memclaw.

Every message (text, photo, voice) goes through the unified MemclawAgent,
which autonomously decides whether to store, search, or just respond.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time

from loguru import logger
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from ..agent import MemclawAgent
from ..backends import AgentBackend
from ..backends.base import ToolStep
from ..config import MemclawConfig
from ..reminders import ReminderScheduler
from . import mask_user_id
from .link_processor import LinkProcessor


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


# Shown in Telegram's command menu (registered in cli._run_telegram).
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

_APPLIES = "Applies from your next message; the conversation is kept."


def model_callback_data(index: int, model_id: str) -> str:
    """``m:<id>``, or ``mi:<index>`` if the id would exceed Telegram's limit."""
    data = f"m:{model_id}"
    return data if len(data.encode()) <= CALLBACK_DATA_MAX_BYTES else f"mi:{index}"


def _unsupported(backend: AgentBackend, what: str) -> str:
    return f"{what} isn't supported for the {backend.display_name} backend yet."


async def _load_models(backend: AgentBackend) -> list:
    try:
        return await backend.list_models()
    except Exception as exc:
        logger.warning("Could not fetch the model list: {exc}", exc=exc)
        return []


def _status_lines(backend: AgentBackend, models: list, levels: list[str] | None) -> list[str]:
    info = next((m for m in models if m.id == backend.model), None)
    model = f"{info.display_name} ({info.id})" if info else backend.model
    if levels == []:
        effort = "not supported by this model"
    else:
        effort = backend.effort or "model default"
    return [f"Model: {model}", f"Effort: {effort}"]


async def model_menu(
    backend: AgentBackend, note: str = "",
) -> tuple[str, InlineKeyboardMarkup | None]:
    """`/model` reply: current settings plus one button per model."""
    if not backend.supports_model_selection:
        return _unsupported(backend, "Switching models"), None
    models = await _load_models(backend)
    lines = _status_lines(backend, models, await backend.effort_levels())
    if note:
        lines += ["", note]
    lines.append("")
    if not models:
        lines.append("Couldn't load the model list right now; send /model <id> to switch.")
        return "\n".join(lines), None
    lines.append("Tap a model to switch, or send /model <id>.")
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            ("✓ " if m.id == backend.model else "") + m.display_name,
            callback_data=model_callback_data(i, m.id),
        )]
        for i, m in enumerate(models)
    ])
    return "\n".join(lines), keyboard


async def effort_menu(
    backend: AgentBackend, note: str = "",
) -> tuple[str, InlineKeyboardMarkup | None]:
    """`/effort` reply: current settings plus a row of level buttons."""
    if not backend.supports_model_selection:
        return _unsupported(backend, "Changing the effort"), None
    levels = await backend.effort_levels()
    lines = _status_lines(backend, await _load_models(backend), levels)
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


async def select_model(backend: AgentBackend, model_id: str) -> str:
    """Switch to *model_id*. Returns a note for the user; raises ValueError
    for an unknown model."""
    if model_id == backend.model:
        return f"Already using {model_id}."
    note = await backend.set_model(model_id)
    return " ".join(p for p in (f"Switched to {backend.model}.", note, _APPLIES) if p)


async def select_effort(backend: AgentBackend, level: str) -> str:
    """Set the effort level. Returns a note; raises ValueError if rejected."""
    await backend.set_effort(level)
    return f"Effort set to {backend.effort}. {_APPLIES}"


async def _resolve_model_arg(backend: AgentBackend, arg: str) -> str:
    """`/model <number or id>` → model id. Raises ValueError."""
    if not arg.isdigit():
        return arg
    models = await _load_models(backend)
    n = int(arg)
    if not 1 <= n <= len(models):
        raise ValueError(f"There's no model number {n}.")
    return models[n - 1].id


class MessageHandlers:
    """Routes every Telegram message through the unified Memclaw agent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.scheduler = ReminderScheduler(config)
        self.agent = MemclawAgent(config, platform="telegram", scheduler=self.scheduler)
        self.link_processor = LinkProcessor(openai_client)
        self._bot = None

    def attach_bot(self, bot):
        """Give the handler a reference to the Telegram Bot for reminder delivery."""
        self._bot = bot
        self.scheduler.register_delivery("telegram", self._deliver_reminder)

    async def _deliver_reminder(self, chat_id: str, text: str):
        if self._bot is None:
            return
        await self._bot.send_message(chat_id=int(chat_id), text=text)
        self.agent.record_reminder_fired(text, chat_id)

    def _check_user(self, user_id: int) -> bool:
        return user_id in self.config.allowed_user_ids_list

    async def _send_response(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        response_text: str,
        found_images: list[dict],
    ):
        """Send agent response: images first, then text."""
        for img in found_images:
            try:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=img["file_id"],
                    caption=img.get("caption") or None,
                )
            except Exception as e:
                logger.error(f"Failed to send image {img.get('file_id')}: {e}")

        if response_text:
            try:
                await update.message.reply_text(response_text[:4096], parse_mode="Markdown")
            except Exception as e:
                # Fall back to plain text if the agent emitted something Telegram's
                # legacy Markdown parser rejects (unbalanced * or _, etc.).
                logger.warning(f"Markdown parse failed, sending plain: {e}")
                await update.message.reply_text(response_text[:4096])

    async def _send_with_typing(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
    ):
        """Run the agent with a typing indicator, then send the full response."""
        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id))
        # One silent status message, edited as the agent works and deleted
        # once the reply is ready.
        status = None
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
            response_text, found_images = await self.agent.handle(
                prompt,
                image_b64=image_b64,
                image_media_type=image_media_type,
                chat_id=str(chat_id),
                on_tool=on_tool,
            )
        finally:
            typing_task.cancel()
            if status is not None:
                with contextlib.suppress(TelegramError):
                    await status.delete()
        await self._send_response(update, context, response_text, found_images)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        await update.message.reply_text(
            "Hi! I'm your personal memory assistant powered by Memclaw.\n\n"
            "Just send me anything — text, photos, or voice messages.\n\n"
            "I'll automatically decide whether to remember it, search your "
            "memories, or retrieve images. No commands needed, just talk to me.\n\n"
            "/new - start a fresh conversation (your memories are kept)\n"
            "/model - show or change the model\n"
            "/effort - show or change the reasoning effort"
        )

    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return
        await self.agent.reset_conversation(str(update.effective_chat.id))
        await update.message.reply_text("Started a new conversation.")

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return
        backend = self.agent.backend
        arg = " ".join(context.args or []).strip()
        note = ""
        if arg and backend.supports_model_selection:
            try:
                note = await select_model(backend, await _resolve_model_arg(backend, arg))
            except ValueError as exc:
                note = f"{exc} Pick one below."
        text, keyboard = await model_menu(backend, note)
        await update.message.reply_text(text, reply_markup=keyboard)

    async def effort_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return
        backend = self.agent.backend
        arg = " ".join(context.args or []).strip()
        note = ""
        if arg and backend.supports_model_selection:
            try:
                note = await select_effort(backend, arg)
            except ValueError as exc:
                note = str(exc)
        text, keyboard = await effort_menu(backend, note)
        await update.message.reply_text(text, reply_markup=keyboard)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Button presses on the /model and /effort menus."""
        query = update.callback_query
        if query is None:
            return
        if not self._check_user(update.effective_user.id):
            await query.answer("You're not allowed to use this bot.")
            return
        backend = self.agent.backend
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
                    models = await _load_models(backend)
                    if not (value.isdigit() and int(value) < len(models)):
                        raise ValueError("That model is no longer available; send /model again.")
                    model_id = models[int(value)].id
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

    # ------------------------------------------------------------------
    # Message handlers — everything goes through the agent
    # ------------------------------------------------------------------

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        text = update.message.text
        logger.info(f"Text from user {mask_user_id(update.effective_user.id)}: {text[:100]}")

        prompt_parts: list[str] = []
        replied = update.message.reply_to_message
        if replied is not None:
            quoted = replied.text or replied.caption or ""
            if quoted:
                prompt_parts.append(f"[Replying to message] {quoted}")

        prompt_parts.append(text)

        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                prompt_parts.append(
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                )

        prompt = "\n".join(prompt_parts)
        await self._send_with_typing(update, context, prompt)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        photo = update.message.photo[-1]
        caption = update.message.caption or ""

        logger.info(f"Photo from user {mask_user_id(update.effective_user.id)}, caption={caption!r}")

        # Download photo and base64 encode
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        logger.debug(f"Downloaded photo: {len(photo_bytes)} bytes")

        # Process links in caption
        link_info = ""
        if caption:
            links = await self.link_processor.process_links(caption)
            for link in links:
                if link.get("summary"):
                    link_info += (
                        f"\n[Link summary] {link['url']}: {link['summary']}"
                        "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                    )

        prompt_text = f"User sent a photo. media_ref={photo.file_id}"
        if caption:
            prompt_text += f"\nCaption: {caption}"
        if link_info:
            prompt_text += link_info

        await self._send_with_typing(
            update, context, prompt_text,
            image_b64=base64_image, image_media_type="image/jpeg",
        )

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        voice = update.message.voice
        logger.info(f"Voice from user {mask_user_id(update.effective_user.id)}")

        # Download and transcribe
        file = await context.bot.get_file(voice.file_id)
        voice_bytes = await file.download_as_bytearray()

        try:
            transcription = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("voice.ogg", bytes(voice_bytes), "audio/ogg"),
            )
        except Exception as exc:
            logger.exception("Whisper transcription failed: {exc}", exc=exc)
            await update.message.reply_text(
                "I couldn't transcribe that voice message. "
                "If you set up Memclaw recently, run `memclaw doctor` to check "
                "your OpenAI key has access to the whisper-1 model."
            )
            return
        text = transcription.text
        logger.info(f"Transcribed: {text}")

        # Process links
        link_info = ""
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                link_info += (
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                )

        # Send to agent so it can respond — transcription is NOT pre-saved
        prompt = (
            f"[Voice message] {text}"
            "\nThis transcription has NOT been saved yet. Save it if the content is worth remembering."
            f"{link_info}"
        )
        await self._send_with_typing(update, context, prompt)

    async def aclose(self):
        await self.agent.aclose()
        self.scheduler.close()

    def close(self):
        self.scheduler.close()
        self.agent.close()
