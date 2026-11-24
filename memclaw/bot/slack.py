"""Slack: receive messages and files, download them, send replies.

Uses slack-bolt with Socket Mode — no public URL required. What happens to
a message in between is shared with the other platforms (memclaw.bot.inbound).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from loguru import logger
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from ..config import MemclawConfig
from . import allowlist_label, console, log_to, mask_user_id, require_setting
from .inbound import TRANSCRIPTION_FAILED, Inbound, is_allowed, save_media

# Image MIME types we handle
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
}


@dataclass
class _Thread:
    """Who sent a message, where, and how to answer in its thread."""

    user: str
    channel: str
    ts: str
    say: Callable[..., Awaitable[Any]]

    async def reply(self, text: str) -> None:
        await self.say(text=text, thread_ts=self.ts)


class SlackBot:
    """Receives Slack messages and files, and sends the replies."""

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.inbound = Inbound(config, "slack")

        self.app = AsyncApp(token=config.slack.bot_token)
        self._register_handlers()
        self.inbound.deliver_reminders(self._send_reminder)

    async def _send_reminder(self, chat_id: str, text: str) -> None:
        await self.app.client.chat_postMessage(channel=chat_id, text=text)

    def _register_handlers(self):
        """Register Slack event handlers on the bolt app."""

        @self.app.event("message")
        async def handle_message_event(event, say, client):
            await self._route_event(event, say)

        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._route_event(event, say)

    # ── Access control ───────────────────────────────────────────────
    # Only the users in SLACK_ALLOWED_USERS (empty lets nobody in), in the
    # channels in SLACK_ALLOWED_CHANNELS (empty = any channel, DMs included).

    def _check_channel(self, channel: str) -> bool:
        channels = self.config.slack.allowed_channels_list
        return not channels or is_allowed(channel, channels)

    def _check_user(self, user: str) -> bool:
        return is_allowed(user, self.config.slack.allowed_users_list)

    # ── Event routing ────────────────────────────────────────────────

    async def _route_event(self, event: dict, say):
        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = event.get("channel", "")
        if not self._check_channel(channel):
            logger.info("Ignoring Slack message in channel {c} (not in SLACK_ALLOWED_CHANNELS)", c=channel)
            return

        user = event.get("user", "unknown")
        if not self._check_user(user):
            logger.debug("Ignoring Slack event from unauthorized user {u}", u=user)
            return
        thread = _Thread(user, channel, event.get("thread_ts") or event.get("ts", ""), say)
        files = event.get("files", [])

        # Strip bot mention from text (e.g. "<@U123ABC> hello" → "hello")
        text = self._strip_mention(event.get("text", ""))

        # Check if there are image files attached
        image_files = [f for f in files if f.get("mimetype", "") in _IMAGE_MIMES]
        audio_files = [f for f in files if f.get("mimetype", "").startswith("audio/")]

        if not files and text.lower() == "/new":
            await thread.reply(await self.inbound.new_conversation(channel))
        elif image_files:
            await self._handle_image(thread, text, image_files[0])
        elif audio_files:
            await self._handle_audio(thread, audio_files[0])
        elif text:
            await self._handle_text(thread, text)
        else:
            logger.debug("Ignoring Slack event with no text or supported files")

    @staticmethod
    def _strip_mention(text: str) -> str:
        """Remove bot mention tags like <@U123ABC> from the start of messages."""
        return re.sub(r"^\s*<@[A-Z0-9]+>\s*", "", text).strip()

    # ── Messages ─────────────────────────────────────────────────────

    async def _handle_text(self, thread: _Thread, text: str):
        logger.info(
            "Slack text from {u} in {c}: {t}", u=mask_user_id(thread.user), c=thread.channel, t=text[:100],
        )
        prompt = await self.inbound.text_prompt(text)
        await self._answer(thread, prompt)

    async def _handle_image(self, thread: _Thread, caption: str, file_info: dict):
        file_name = file_info.get("name", "image")
        logger.info(
            "Slack image from {u}: {f}, caption={c!r}", u=mask_user_id(thread.user), f=file_name, c=caption,
        )

        # Download image via Slack file URL (requires bot token for auth)
        image_bytes = await self._download_slack_file(file_info)
        if image_bytes is None:
            await thread.reply("Sorry, I couldn't download that image.")
            return

        # Save locally under the slack media dir so retrievals can upload it back.
        mime = file_info.get("mimetype", "image/jpeg")
        local_path = save_media(self.config.slack_media_dir, image_bytes, _MIME_TO_EXT.get(mime) or ".jpg")
        logger.debug("Downloaded Slack image: {n} bytes -> {p}", n=len(image_bytes), p=local_path)

        prompt = await self.inbound.photo_prompt(str(local_path), caption)
        media_type = mime if mime.startswith("image/") else "image/jpeg"
        await self._answer(thread, prompt, image=(image_bytes, media_type))

    async def _handle_audio(self, thread: _Thread, file_info: dict):
        file_name = file_info.get("name", "audio")
        logger.info("Slack audio from {u}: {f}", u=mask_user_id(thread.user), f=file_name)

        audio_bytes = await self._download_slack_file(file_info)
        if audio_bytes is None:
            await thread.reply("Sorry, I couldn't download that audio file.")
            return

        mime = file_info.get("mimetype", "audio/mp4")
        ext = _MIME_TO_EXT.get(mime) or ".m4a"
        text = await self.inbound.transcribe(audio_bytes, filename=f"audio{ext}", mime_type=mime)
        if text is None:
            await thread.reply(TRANSCRIPTION_FAILED)
            return
        logger.info("Transcribed Slack audio: {t}", t=text)

        await self._answer(thread, await self.inbound.voice_prompt(text))

    async def _answer(self, thread: _Thread, prompt: str, *, image: tuple[bytes, str] | None = None):
        response_text, found_images = await self.inbound.ask(prompt, chat_id=thread.channel, image=image)
        await self._send_response(thread, response_text, found_images)

    # ── Slack API helpers ────────────────────────────────────────────

    async def _download_slack_file(self, file_info: dict) -> bytes | None:
        """Download a file from Slack using the url_private_download URL."""
        url = file_info.get("url_private_download") or file_info.get("url_private", "")
        if not url:
            logger.error("No download URL in Slack file info")
            return None

        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.config.slack.bot_token}"},
                timeout=30.0,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            logger.error("Failed to download Slack file: {exc}", exc=exc)
            return None

    async def _upload_and_share_image(self, thread: _Thread, image_path: str, caption: str | None = None):
        """Upload a local image to Slack and share it in a channel."""
        path = Path(image_path)
        if not path.exists():
            logger.error("Image file not found for Slack upload: {p}", p=image_path)
            return

        try:
            result = await self.app.client.files_upload_v2(
                channel=thread.channel,
                file=str(path),
                filename=path.name,
                initial_comment=caption or "",
                thread_ts=thread.ts if thread.ts else None,
            )
            logger.debug("Uploaded image to Slack: {r}", r=result.get("ok"))
        except Exception as exc:
            logger.error("Failed to upload image to Slack: {exc}", exc=exc)

    async def _send_response(self, thread: _Thread, response_text: str, found_images: list[dict]):
        """Send agent response: images first, then text."""
        for img in found_images:
            platform = img.get("platform", "telegram")
            media_ref = img.get("media_ref") or img.get("file_id", "")
            caption = img.get("caption") or None

            if platform == "slack":
                await self._upload_and_share_image(thread, media_ref, caption)
            else:
                # For images from other platforms (e.g. Telegram file_ids),
                # include a note in the text response
                desc = img.get("description", "an image")
                if response_text:
                    response_text += f"\n\n_(Found image: {desc} -- originally saved via {platform})_"
                else:
                    response_text = f"_(Found image: {desc} -- originally saved via {platform})_"

        if response_text:
            await thread.reply(response_text[:4000])

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        """Start the Slack bot via Socket Mode."""
        await self.inbound.start()
        handler = AsyncSocketModeHandler(self.app, self.config.slack.app_token)
        await handler.start_async()

    async def aclose(self):
        await self.inbound.aclose()


def run(config: MemclawConfig) -> None:
    """Start the Slack bot (Socket Mode) and block until it stops."""
    require_setting(config.slack.bot_token, "SLACK_BOT_TOKEN")
    require_setting(config.slack.app_token, "SLACK_APP_TOKEN")

    log_to(config.memory_dir / "slack.log")

    bot = SlackBot(config)

    users = allowlist_label([mask_user_id(u) for u in config.slack.allowed_users_list], "SLACK_ALLOWED_USERS")
    channels = ", ".join(config.slack.allowed_channels_list) or "any"
    console.print(
        f"[green]Starting Memclaw Slack bot (Socket Mode)...[/green]  "
        f"(allowed users: {users}; allowed channels: {channels})"
    )

    async def _run():
        try:
            await bot.start()
        except KeyboardInterrupt:
            pass
        finally:
            await bot.aclose()
            console.print("\nMemclaw Slack bot shut down.")

    asyncio.run(_run())
