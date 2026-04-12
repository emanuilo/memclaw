"""WhatsApp bot handlers for Memclaw (personal account via WhatsApp Web).

Uses `neonize` (whatsmeow Go library under the hood) to connect to the user's
personal WhatsApp account via QR code pairing — no Meta Business account,
no webhooks, no public server.

Every incoming message (text, image, voice) goes through the unified
MemclawAgent, which decides whether to store, search, or just respond.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from neonize.aioze.client import NewAClient
from neonize.aioze.events import ConnectedEv, MessageEv, PairStatusEv

from ..agent import MemclawAgent
from ..config import MemclawConfig
from .link_processor import LinkProcessor


class WhatsAppBot:
    """Personal WhatsApp bot backed by neonize + MemclawAgent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.agent = MemclawAgent(config)
        self.link_processor = LinkProcessor(openai_client)

        self.client = NewAClient(str(config.whatsapp_session_db))
        self._register_handlers()

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def _register_handlers(self):
        @self.client.event(ConnectedEv)
        async def _on_connected(_cli: NewAClient, _ev: ConnectedEv):
            logger.info("WhatsApp connected")

        @self.client.event(PairStatusEv)
        async def _on_pair(_cli: NewAClient, ev: PairStatusEv):
            logger.info("WhatsApp paired as +{jid}", jid=ev.ID.User)

        @self.client.event(MessageEv)
        async def _on_message(cli: NewAClient, ev: MessageEv):
            try:
                await self._route_message(cli, ev)
            except Exception as exc:
                logger.exception("Error handling WhatsApp message: {exc}", exc=exc)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_sender(self, ev: MessageEv) -> bool:
        source = ev.Info.MessageSource

        # Ignore group messages by default — this is a personal notes bot
        if source.IsGroup:
            return False

        # Messages from the paired account (self-notes) are always allowed
        if source.IsFromMe:
            return True

        allowed = self.config.allowed_whatsapp_numbers_list
        if not allowed:
            # No allowlist: only self-notes are allowed
            return False

        sender_phone = source.Sender.User
        return any(a.lstrip("+") == sender_phone for a in allowed)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _route_message(self, cli: NewAClient, ev: MessageEv):
        if not self._check_sender(ev):
            return

        msg = ev.Message

        if msg.imageMessage.ListFields():
            await self._handle_image(cli, ev)
            return
        if msg.audioMessage.ListFields():
            await self._handle_audio(cli, ev)
            return

        text = msg.conversation or msg.extendedTextMessage.text
        if text:
            await self._handle_text(cli, ev, text)
        else:
            logger.debug("Ignoring unsupported WhatsApp message type")

    # ------------------------------------------------------------------
    # Text messages
    # ------------------------------------------------------------------

    async def _handle_text(self, cli: NewAClient, ev: MessageEv, text: str):
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp text from {s}: {t}", s=sender, t=text[:100])

        prompt_parts = [text]
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                prompt_parts.append(
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                )

        prompt = "\n".join(prompt_parts)
        response_text, found_images = await self.agent.handle(prompt)
        await self._send_response(cli, ev, response_text, found_images)

    # ------------------------------------------------------------------
    # Image messages
    # ------------------------------------------------------------------

    async def _handle_image(self, cli: NewAClient, ev: MessageEv):
        img_msg = ev.Message.imageMessage
        caption = img_msg.caption or ""
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp image from {s}, caption={c!r}", s=sender, c=caption)

        try:
            image_bytes = await cli.download_any(ev.Message)
        except Exception as exc:
            logger.error("Failed to download WhatsApp image: {exc}", exc=exc)
            await cli.reply_message("Sorry, I couldn't download that image.", ev)
            return

        mime_type = img_msg.mimetype or "image/jpeg"
        ext = _mime_to_ext(mime_type) or ".jpg"
        local_path = self.config.whatsapp_media_dir / f"{uuid.uuid4().hex}{ext}"
        local_path.write_bytes(image_bytes)

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        link_info = ""
        if caption:
            links = await self.link_processor.process_links(caption)
            for link in links:
                if link.get("summary"):
                    link_info += (
                        f"\n[Link summary] {link['url']}: {link['summary']}"
                        "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                    )

        prompt_text = f"User sent a photo. file_path={local_path}"
        if caption:
            prompt_text += f"\nCaption: {caption}"
        if link_info:
            prompt_text += link_info

        media_type = mime_type if mime_type.startswith("image/") else "image/jpeg"
        response_text, found_images = await self.agent.handle(
            prompt_text, image_b64=base64_image, image_media_type=media_type,
        )
        await self._send_response(cli, ev, response_text, found_images)

    # ------------------------------------------------------------------
    # Audio / voice messages
    # ------------------------------------------------------------------

    async def _handle_audio(self, cli: NewAClient, ev: MessageEv):
        audio_msg = ev.Message.audioMessage
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp voice/audio from {s}", s=sender)

        try:
            audio_bytes = await cli.download_any(ev.Message)
        except Exception as exc:
            logger.error("Failed to download WhatsApp audio: {exc}", exc=exc)
            await cli.reply_message("Sorry, I couldn't download that audio message.", ev)
            return

        mime_type = audio_msg.mimetype or "audio/ogg"
        ext = _mime_to_ext(mime_type) or ".ogg"

        transcription = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"voice{ext}", audio_bytes, mime_type),
        )
        text = transcription.text
        logger.debug("Transcribed WhatsApp voice: {t}", t=text[:100])

        link_info = ""
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                link_info += (
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                )

        prompt = (
            f"[Voice message] {text}"
            "\nThis transcription has NOT been saved yet. Save it if the content is worth remembering."
            f"{link_info}"
        )
        response_text, found_images = await self.agent.handle(prompt)
        await self._send_response(cli, ev, response_text, found_images)

    # ------------------------------------------------------------------
    # Sending replies
    # ------------------------------------------------------------------

    async def _send_response(
        self,
        cli: NewAClient,
        ev: MessageEv,
        response_text: str,
        found_images: list[dict],
    ):
        """Send agent response: images first, then text."""
        chat = ev.Info.MessageSource.Chat

        for img in found_images:
            platform = img.get("platform", "telegram")
            media_ref = img.get("media_ref") or img.get("file_id", "")
            caption = img.get("caption") or None

            if platform == "whatsapp" and media_ref and Path(media_ref).exists():
                try:
                    await cli.send_image(chat, media_ref, caption=caption)
                except Exception as exc:
                    logger.error("Failed to send WhatsApp image: {exc}", exc=exc)
            else:
                desc = img.get("description", "an image")
                note = f"(Found image: {desc} — originally saved via {platform})"
                response_text = f"{response_text}\n\n{note}" if response_text else note

        if response_text:
            try:
                await cli.send_message(chat, response_text)
            except Exception as exc:
                logger.error("Failed to send WhatsApp message: {exc}", exc=exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Connect to WhatsApp. On first run, a QR code is printed to stdout."""
        await self.agent.start()
        await self.agent.start_background_sync(interval=60)
        await self.client.connect()
        await self.client.idle()

    def close(self):
        self.agent.close()


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/ogg": ".ogg",
    "audio/ogg; codecs=opus": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
}


def _mime_to_ext(mime_type: str) -> str:
    return _MIME_TO_EXT.get(mime_type, _MIME_TO_EXT.get(mime_type.split(";")[0].strip(), ""))
