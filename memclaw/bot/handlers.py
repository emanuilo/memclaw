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
from ..config import MemclawConfig
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


class MessageHandlers:
    """Routes every Telegram message through the unified Memclaw agent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.agent = MemclawAgent(config, platform="telegram")
        self.link_processor = LinkProcessor(openai_client)

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
                prompt, image_b64=image_b64, image_media_type=image_media_type
            )
        finally:
            typing_task.cancel()
        await self._send_response(update, context, response_text, found_images)

    # ------------------------------------------------------------------
    # /start — the only command
    # ------------------------------------------------------------------

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        await update.message.reply_text(
            "Hi! I'm your personal memory assistant powered by Memclaw.\n\n"
            "Just send me anything — text, photos, or voice messages.\n\n"
            "I'll automatically decide whether to remember it, search your "
            "memories, or retrieve images. No commands needed, just talk to me."
        )

    # ------------------------------------------------------------------
    # Message handlers — everything goes through the agent
    # ------------------------------------------------------------------

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        text = update.message.text
        logger.info(f"Text from user {update.effective_user.id}: {text[:100]}")

        # Pre-process links and include summaries in the prompt
        prompt_parts = [text]
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

        logger.info(f"Photo from user {update.effective_user.id}, caption={caption!r}")

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
        logger.info(f"Voice from user {update.effective_user.id}")

        # Download and transcribe
        file = await context.bot.get_file(voice.file_id)
        voice_bytes = await file.download_as_bytearray()

        transcription = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=("voice.ogg", bytes(voice_bytes), "audio/ogg"),
        )
        text = transcription.text
        logger.debug(f"Transcribed: {text[:100]}")

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

    def close(self):
        self.agent.close()
