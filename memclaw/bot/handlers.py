"""Telegram bot message handlers for Memclaw.

Every message (text, photo, voice) goes through the unified MemclawAgent,
which autonomously decides whether to store, search, or just respond.
"""

from __future__ import annotations

import asyncio
import base64

from loguru import logger
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from ..agent import MemclawAgent
from ..backends import AgentBackend
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


async def model_command_reply(backend: AgentBackend, args: list[str]) -> str:
    """Reply for `/model` (list models) or `/model <number or id>` (switch)."""
    if not backend.supports_model_selection:
        return f"Switching models isn't supported for the {backend.display_name} backend yet."
    try:
        models = await backend.list_models()
    except Exception as exc:
        logger.warning("Could not fetch the model list: {exc}", exc=exc)
        models = []

    arg = " ".join(args).strip()
    if arg:
        model_id = arg
        if arg.isdigit():
            n = int(arg)
            if not 1 <= n <= len(models):
                return f"There's no model number {n}. Send /model to see the list."
            model_id = models[n - 1].id
        try:
            note = await backend.set_model(model_id)
        except ValueError as exc:
            return f"{exc} Send /model to see the available models."
        lines = [f"Switched to {backend.model}."]
        if note:
            lines.append(note)
        lines.append(f"Effort: {backend.effort or 'model default'}")
        lines.append("Applies from your next message; the conversation is kept.")
        return "\n".join(lines)

    lines = [
        f"Model: {backend.model}",
        f"Effort: {backend.effort or 'model default'}",
        "",
    ]
    if models:
        lines.append("Available models:")
        for i, m in enumerate(models, 1):
            current = "  (current)" if m.id == backend.model else ""
            lines.append(f"{i}. {m.display_name} - {m.id}{current}")
        lines += ["", "Send /model <number or id> to switch."]
    else:
        lines.append("Couldn't load the model list right now; send /model <id> to switch.")
    return "\n".join(lines)


async def effort_command_reply(backend: AgentBackend, args: list[str]) -> str:
    """Reply for `/effort` (show levels) or `/effort <level>` (set it)."""
    if not backend.supports_model_selection:
        return f"Changing the effort isn't supported for the {backend.display_name} backend yet."
    arg = " ".join(args).strip()
    if arg:
        try:
            await backend.set_effort(arg)
        except ValueError as exc:
            return str(exc)
        return (
            f"Effort set to {backend.effort}.\n"
            "Applies from your next message; the conversation is kept."
        )

    levels = await backend.effort_levels()
    lines = [
        f"Model: {backend.model}",
        f"Effort: {backend.effort or 'model default'}",
        "",
    ]
    if levels == []:
        lines.append("This model doesn't support an effort setting.")
    else:
        if levels:
            lines.append(f"Levels this model supports: {', '.join(levels)}")
        lines.append("Send /effort <level> to change it.")
    return "\n".join(lines)


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
        try:
            response_text, found_images = await self.agent.handle(
                prompt,
                image_b64=image_b64,
                image_media_type=image_media_type,
                chat_id=str(chat_id),
            )
        finally:
            typing_task.cancel()
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
        reply = await model_command_reply(self.agent.backend, list(context.args or []))
        await update.message.reply_text(reply)

    async def effort_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return
        reply = await effort_command_reply(self.agent.backend, list(context.args or []))
        await update.message.reply_text(reply)

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
