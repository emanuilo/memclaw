"""Telegram bot message handlers for Memclaw.

Handles incoming messages (text, photo, voice) and commands.
Silently stores all incoming content as memories; uses the Claude Agent SDK
for the /ask command.
"""

from __future__ import annotations

import base64
from functools import wraps
from pathlib import Path

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


def restricted(allowed_user_ids: list[int]):
    """Decorator that silently drops messages from unauthorised users."""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_user.id not in allowed_user_ids:
                return
            return await func(self, update, context)

        return wrapper

    return decorator


class MessageHandlers:
    """Handles all Telegram message types and bot commands."""

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

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        await update.message.reply_text(
            "Hi! I'm your personal memory assistant powered by Memclaw.\n\n"
            "Send me text, images, or voice messages — I'll remember everything.\n\n"
            "Commands:\n"
            "/ask <question> - Ask me anything based on your memories\n"
            "/search <query> - Search your memories\n"
            "/memories - Show today's entries\n"
            "/stats - Show statistics"
        )

    async def ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask — uses Claude Agent SDK with memory context."""
        if not self._check_user(update.effective_user.id):
            return

        question = " ".join(context.args) if context.args else ""
        if not question:
            await update.message.reply_text("Usage: /ask <your question>")
            return

        logger.info(f"/ask from user {update.effective_user.id}: {question}")

        response_text, found_images = await self.agent.ask(question)

        # Send images first
        image_sent = False
        for img in found_images:
            try:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=img["file_id"],
                    caption=img.get("caption") or None,
                )
                image_sent = True
            except Exception as e:
                logger.error(f"Failed to send image {img.get('file_id')}: {e}")

        # Always send the text response
        if response_text:
            await update.message.reply_text(response_text[:4096])

    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        query_text = " ".join(context.args) if context.args else ""
        if not query_text:
            await update.message.reply_text("Usage: /search <query>")
            return

        results = await self.search.search(query_text, limit=5)
        if not results:
            await update.message.reply_text("No relevant memories found.")
            return

        response = "Search Results:\n\n"
        for i, r in enumerate(results, 1):
            source = Path(r.file_path).stem
            response += f"{i}. [{source}] {r.content.strip()[:200]}\n\n"

        await update.message.reply_text(response[:4096])

    async def memories_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        files = self.store.list_files()
        stats = self.index.get_stats()

        response = f"Memory files: {len(files)}  |  Chunks: {stats['chunks']}  |  Images: {stats['images']}\n\n"

        today_file = self.config.daily_file()
        if today_file.exists():
            response += f"Today's entries:\n{self.store.read_file(today_file)[:3000]}"
        else:
            response += "No entries today yet."

        await update.message.reply_text(response[:4096])

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        files = self.store.list_files()
        stats = self.index.get_stats()

        await update.message.reply_text(
            f"Memclaw Statistics:\n\n"
            f"Memory files: {len(files)}\n"
            f"Indexed chunks: {stats['chunks']}\n"
            f"Stored images: {stats['images']}\n"
            f"Memory directory: {self.config.memory_dir}"
        )

    # ------------------------------------------------------------------
    # Message handlers — silent storage
    # ------------------------------------------------------------------

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        text = update.message.text
        logger.info(f"Text from user {update.effective_user.id}: {text[:100]}")

        # Process links
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                link_entry = f"Link: {link['url']}\nSummary: {link['summary']}"
                file_path = self.store.save(link_entry, entry_type="link")
                await self.index.index_file(file_path)

        # Store the message itself
        file_path = self.store.save(text, entry_type="note")
        await self.index.index_file(file_path)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update.effective_user.id):
            return

        photo = update.message.photo[-1]
        caption = update.message.caption or ""

        logger.info(f"Photo from user {update.effective_user.id}")

        # Download and encode
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")

        # Generate description via vision API
        prompt = "Describe this image in detail in about 50 tokens."
        if caption:
            prompt += f" Caption: {caption}"

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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

        # Store in memory as text
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
        if caption:
            links = await self.link_processor.process_links(caption)
            for link in links:
                if link.get("summary"):
                    lp = self.store.save(
                        f"Link: {link['url']}\nSummary: {link['summary']}",
                        entry_type="link",
                    )
                    await self.index.index_file(lp)

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

        # Process links from transcription
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                lp = self.store.save(
                    f"Link: {link['url']}\nSummary: {link['summary']}",
                    entry_type="link",
                )
                await self.index.index_file(lp)

    def close(self):
        self.index.close()
