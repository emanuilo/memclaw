"""WhatsApp (your personal account via WhatsApp Web): receive messages,
download their media, send replies.

Uses `neonize` (whatsmeow Go library under the hood) to connect to the user's
personal WhatsApp account via QR code pairing — no Meta Business account,
no webhooks, no public server. Only your own notes to yourself are handled.
What happens to a message in between is shared with the other platforms
(memclaw.bot.inbound).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from neonize.aioze.client import NewAClient
from neonize.aioze.events import ConnectedEv, MessageEv, PairStatusEv
from neonize.utils.jid import build_jid

from ..config import MemclawConfig
from . import console, log_to, mask_user_id, run_until_stopped
from .inbound import TRANSCRIPTION_FAILED, Inbound, save_media

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


class WhatsAppBot:
    """Personal WhatsApp bot backed by neonize and the shared pipeline."""

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.inbound = Inbound(config, "whatsapp")

        self.client = NewAClient(str(config.whatsapp_session_db))
        self._register_handlers()
        self.inbound.deliver_reminders(self._send_reminder)

    async def _send_reminder(self, chat_id: str, text: str) -> None:
        """chat_id is stored as 'user@server' (e.g. '12345@s.whatsapp.net')."""
        if "@" in chat_id:
            user, server = chat_id.split("@", 1)
        else:
            user, server = chat_id, "s.whatsapp.net"
        await self.client.send_message(build_jid(user, server=server), text)

    def _register_handlers(self):
        @self.client.event(ConnectedEv)
        async def _on_connected(_cli: NewAClient, _ev: ConnectedEv):
            logger.info("WhatsApp connected")

        @self.client.event(PairStatusEv)
        async def _on_pair(_cli: NewAClient, ev: PairStatusEv):
            logger.info("WhatsApp paired as +{jid}", jid=mask_user_id(ev.ID.User))

        @self.client.event(MessageEv)
        async def _on_message(cli: NewAClient, ev: MessageEv):
            try:
                await self._route_message(cli, ev)
            except Exception as exc:
                logger.exception("Error handling WhatsApp message: {exc}", exc=exc)

    # ── Access control ───────────────────────────────────────────────

    def _check_sender(self, ev: MessageEv) -> bool:
        """Only process self-notes (your own messages to yourself).

        IsFromMe alone is not enough — it also matches outgoing DMs to other
        people. The chat JID must equal the sender JID, which is only true in
        the self-chat.
        """
        source = ev.Info.MessageSource
        if source.IsGroup:
            return False
        if not source.IsFromMe:
            return False
        return source.Chat.User == source.Sender.User

    # ── Message routing ──────────────────────────────────────────────

    @staticmethod
    def _chat_id(ev: MessageEv) -> str:
        chat = ev.Info.MessageSource.Chat
        return f"{chat.User}@{chat.Server}"

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
        if text and text.strip().lower() == "/new":
            reply = await self.inbound.new_conversation(self._chat_id(ev))
            await cli.send_message(ev.Info.MessageSource.Chat, reply)
        elif text:
            await self._handle_text(cli, ev, text)
        else:
            logger.debug("Ignoring unsupported WhatsApp message type")

    # ── Messages ─────────────────────────────────────────────────────

    async def _handle_text(self, cli: NewAClient, ev: MessageEv, text: str):
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp text from {s}: {t}", s=mask_user_id(sender), t=text[:100])

        await self._answer(cli, ev, await self.inbound.text_prompt(text))

    async def _handle_image(self, cli: NewAClient, ev: MessageEv):
        img_msg = ev.Message.imageMessage
        caption = img_msg.caption or ""
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp image from {s}, caption={c!r}", s=mask_user_id(sender), c=caption)

        try:
            image_bytes = await cli.download_any(ev.Message)
        except Exception as exc:
            logger.exception("Failed to download WhatsApp image: {exc}", exc=exc)
            await cli.reply_message("Sorry, I couldn't download that image.", ev)
            return

        mime_type = img_msg.mimetype or "image/jpeg"
        local_path = save_media(self.config.whatsapp_media_dir, image_bytes, _mime_to_ext(mime_type) or ".jpg")

        prompt = await self.inbound.photo_prompt(str(local_path), caption)
        media_type = mime_type if mime_type.startswith("image/") else "image/jpeg"
        await self._answer(cli, ev, prompt, image=(image_bytes, media_type))

    async def _handle_audio(self, cli: NewAClient, ev: MessageEv):
        audio_msg = ev.Message.audioMessage
        sender = ev.Info.MessageSource.Sender.User
        logger.info("WhatsApp voice/audio from {s}", s=mask_user_id(sender))

        try:
            audio_bytes = await cli.download_any(ev.Message)
        except Exception as exc:
            logger.exception("Failed to download WhatsApp audio: {exc}", exc=exc)
            await cli.reply_message("Sorry, I couldn't download that audio message.", ev)
            return

        mime_type = audio_msg.mimetype or "audio/ogg"
        ext = _mime_to_ext(mime_type) or ".ogg"
        text = await self.inbound.transcribe(audio_bytes, filename=f"voice{ext}", mime_type=mime_type)
        if text is None:
            await cli.reply_message(TRANSCRIPTION_FAILED, ev)
            return
        logger.info("Transcribed WhatsApp voice: {t}", t=text)

        await self._answer(cli, ev, await self.inbound.voice_prompt(text))

    async def _answer(
        self, cli: NewAClient, ev: MessageEv, prompt: str, *, image: tuple[bytes, str] | None = None,
    ):
        response_text, found_images = await self.inbound.ask(prompt, chat_id=self._chat_id(ev), image=image)
        await self._send_response(cli, ev, response_text, found_images)

    # ── Sending replies ──────────────────────────────────────────────

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
            note = await self._send_image(cli, chat, img)
            if note:
                response_text = f"{response_text}\n\n{note}" if response_text else note

        if response_text:
            try:
                await cli.send_message(chat, response_text)
            except Exception as exc:
                logger.exception("Failed to send WhatsApp message: {exc}", exc=exc)

    @staticmethod
    async def _send_image(cli: NewAClient, chat, img: dict) -> str:
        """Send an image saved on WhatsApp. For one saved elsewhere, return a
        note to add to the reply instead."""
        platform = img.get("platform", "telegram")
        media_ref = img.get("media_ref") or img.get("file_id", "")
        if not (platform == "whatsapp" and media_ref and Path(media_ref).exists()):
            desc = img.get("description", "an image")
            return f"(Found image: {desc} — originally saved via {platform})"
        try:
            await cli.send_image(chat, media_ref, caption=img.get("caption") or None)
        except Exception as exc:
            logger.exception("Failed to send WhatsApp image: {exc}", exc=exc)
        return ""

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        """Connect to WhatsApp. On first run, a QR code is printed to stdout."""
        await self.inbound.start()
        await self.client.connect()
        await self.client.idle()

    async def aclose(self):
        await self.inbound.aclose()


def run(config: MemclawConfig) -> None:
    """Start the WhatsApp bot and block until it stops."""
    log_to(config.memory_dir / "whatsapp.log")

    bot = WhatsAppBot(config)

    console.print("[green]Starting Memclaw WhatsApp bot[/green]  (self-notes only)")
    if not config.whatsapp_session_db.exists():
        console.print(
            "[cyan]First run:[/cyan] a QR code will appear below. "
            "Open WhatsApp → Settings → Linked Devices → Link a Device, and scan it."
        )

    run_until_stopped(bot.start, bot.aclose, "WhatsApp")
