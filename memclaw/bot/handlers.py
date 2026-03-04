"""Telegram bot message handlers for Memclaw.

Every message (text, photo, voice) goes through the Claude agent, which
autonomously decides whether to store, search, or just respond.
"""

from __future__ import annotations

import base64

from loguru import logger
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import ContextTypes

from ..agent import TelegramAgent
from ..config import MemclawConfig
from ..index import MemoryIndex
from ..search import HybridSearch
from ..store import MemoryStore
from .link_processor import LinkProcessor


class MessageHandlers:
    """Routes every Telegram message through the Claude agent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self.agent = TelegramAgent(config, self.store, self.index, self.search)
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
                # Store link summary in index for future retrieval
                link_entry = f"Link: {link['url']}\nSummary: {link['summary']}"
                file_path = self.store.save(link_entry, entry_type="link")
                await self.index.index_file(file_path)
                prompt_parts.append(
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                )

        prompt = "\n".join(prompt_parts)
        response_text, found_images = await self.agent.handle(prompt)
        await self._send_response(update, context, response_text, found_images)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        photo = update.message.photo[-1]
        caption = update.message.caption or ""

        logger.info(f"Photo from user {update.effective_user.id}")

        # Download and describe via vision API
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")

        vision_prompt = "Describe this image in detail in about 50 tokens."
        if caption:
            vision_prompt += f" Caption: {caption}"

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            max_tokens=100,
        )
        description = response.choices[0].message.content

        # Store image description in memory
        combined = f"Image: {description}."
        if caption:
            combined += f" Caption: {caption}"

        file_path = self.store.save(combined, entry_type="image")
        await self.index.index_file(file_path)

        # Store in Telegram image registry for file_id retrieval
        await self.index.store_telegram_image(
            file_id=photo.file_id,
            description=combined,
            caption=caption,
        )

        # Process links in caption
        link_info = ""
        if caption:
            links = await self.link_processor.process_links(caption)
            for link in links:
                if link.get("summary"):
                    lp = self.store.save(
                        f"Link: {link['url']}\nSummary: {link['summary']}",
                        entry_type="link",
                    )
                    await self.index.index_file(lp)
                    link_info += f"\n[Link summary] {link['url']}: {link['summary']}"

        # Send to agent so it can respond
        prompt = f"[Image received] {combined}{link_info}"
        response_text, found_images = await self.agent.handle(prompt)
        await self._send_response(update, context, response_text, found_images)

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

        # Store transcription
        file_path = self.store.save(text, entry_type="voice")
        await self.index.index_file(file_path)

        # Process links
        link_info = ""
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                lp = self.store.save(
                    f"Link: {link['url']}\nSummary: {link['summary']}",
                    entry_type="link",
                )
                await self.index.index_file(lp)
                link_info += f"\n[Link summary] {link['url']}: {link['summary']}"

        # Send to agent so it can respond
        prompt = f"[Voice message] {text}{link_info}"
        response_text, found_images = await self.agent.handle(prompt)
        await self._send_response(update, context, response_text, found_images)

    def close(self):
        self.index.close()
