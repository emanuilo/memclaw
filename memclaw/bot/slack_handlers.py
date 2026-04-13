"""Slack bot event handlers for Memclaw.

Every message (text, file/image) goes through the unified MemclawAgent,
which autonomously decides whether to store, search, or just respond.

Uses slack-bolt with Socket Mode — no public URL required.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from ..agent import MemclawAgent
from ..config import MemclawConfig
from ..reminders import ReminderScheduler
from .link_processor import LinkProcessor

# Image MIME types we handle
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}


class SlackHandlers:
    """Routes every Slack message through the unified Memclaw agent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.scheduler = ReminderScheduler(config)
        self.agent = MemclawAgent(config, platform="slack", scheduler=self.scheduler)
        self.link_processor = LinkProcessor(openai_client)

        self.app = AsyncApp(token=config.slack_bot_token)
        self._register_handlers()
        self.scheduler.register_delivery("slack", self._deliver_reminder)

    async def _deliver_reminder(self, chat_id: str, text: str):
        await self.app.client.chat_postMessage(channel=chat_id, text=text)

    def _register_handlers(self):
        """Register Slack event handlers on the bolt app."""

        @self.app.event("message")
        async def handle_message_event(event, say, client):
            await self._route_event(event, say, client)

        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._route_event(event, say, client)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_channel(self, channel: str) -> bool:
        allowed = self.config.slack_allowed_channels_list
        if not allowed:
            return True  # no allowlist = allow all
        return channel in allowed

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    async def _route_event(self, event: dict, say, client):
        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        channel = event.get("channel", "")
        if not self._check_channel(channel):
            return

        user = event.get("user", "unknown")
        thread_ts = event.get("thread_ts") or event.get("ts", "")
        text = event.get("text", "")
        files = event.get("files", [])

        # Strip bot mention from text (e.g. "<@U123ABC> hello" → "hello")
        text = self._strip_mention(text)

        # Check if there are image files attached
        image_files = [f for f in files if f.get("mimetype", "") in _IMAGE_MIMES]
        audio_files = [f for f in files if f.get("mimetype", "").startswith("audio/")]

        if image_files:
            await self._handle_image(user, channel, thread_ts, text, image_files[0], say, client)
        elif audio_files:
            await self._handle_audio(user, channel, thread_ts, text, audio_files[0], say, client)
        elif text:
            await self._handle_text(user, channel, thread_ts, text, say)
        else:
            logger.debug("Ignoring Slack event with no text or supported files")

    @staticmethod
    def _strip_mention(text: str) -> str:
        """Remove bot mention tags like <@U123ABC> from the start of messages."""
        import re
        return re.sub(r"^\s*<@[A-Z0-9]+>\s*", "", text).strip()

    # ------------------------------------------------------------------
    # Text messages
    # ------------------------------------------------------------------

    async def _handle_text(self, user: str, channel: str, thread_ts: str, text: str, say):
        logger.info("Slack text from {u} in {c}: {t}", u=user, c=channel, t=text[:100])

        prompt_parts = [text]
        links = await self.link_processor.process_links(text)
        for link in links:
            if link.get("summary"):
                prompt_parts.append(
                    f"\n[Link summary] {link['url']}: {link['summary']}"
                    "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
                )

        prompt = "\n".join(prompt_parts)
        response_text, found_images = await self.agent.handle(prompt, chat_id=channel)
        await self._send_response(channel, thread_ts, response_text, found_images, say)

    # ------------------------------------------------------------------
    # Image messages
    # ------------------------------------------------------------------

    async def _handle_image(
        self, user: str, channel: str, thread_ts: str, caption: str,
        file_info: dict, say, client,
    ):
        file_name = file_info.get("name", "image")
        logger.info("Slack image from {u}: {f}, caption={c!r}", u=user, f=file_name, c=caption)

        # Download image via Slack file URL (requires bot token for auth)
        image_bytes = await self._download_slack_file(file_info)
        if image_bytes is None:
            await say(text="Sorry, I couldn't download that image.", thread_ts=thread_ts)
            return

        # Save locally under the slack media dir so retrievals can upload it back.
        mime = file_info.get("mimetype", "image/jpeg")
        ext = _mime_to_ext(mime) or ".jpg"
        local_path = self.config.slack_media_dir / f"{uuid.uuid4().hex}{ext}"
        local_path.write_bytes(image_bytes)

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        logger.debug("Downloaded Slack image: {n} bytes -> {p}", n=len(image_bytes), p=local_path)

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

        prompt_text = f"User sent a photo. media_ref={local_path}"
        if caption:
            prompt_text += f"\nCaption: {caption}"
        if link_info:
            prompt_text += link_info

        media_type = mime if mime.startswith("image/") else "image/jpeg"
        response_text, found_images = await self.agent.handle(
            prompt_text,
            image_b64=base64_image,
            image_media_type=media_type,
            chat_id=channel,
        )
        await self._send_response(channel, thread_ts, response_text, found_images, say)

    # ------------------------------------------------------------------
    # Audio messages
    # ------------------------------------------------------------------

    async def _handle_audio(
        self, user: str, channel: str, thread_ts: str, caption: str,
        file_info: dict, say, client,
    ):
        file_name = file_info.get("name", "audio")
        logger.info("Slack audio from {u}: {f}", u=user, f=file_name)

        audio_bytes = await self._download_slack_file(file_info)
        if audio_bytes is None:
            await say(text="Sorry, I couldn't download that audio file.", thread_ts=thread_ts)
            return

        mime = file_info.get("mimetype", "audio/mp4")
        ext = _mime_to_ext(mime) or ".m4a"

        transcription = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio{ext}", audio_bytes, mime),
        )
        text = transcription.text
        logger.debug("Transcribed Slack audio: {t}", t=text[:100])

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
        response_text, found_images = await self.agent.handle(prompt, chat_id=channel)
        await self._send_response(channel, thread_ts, response_text, found_images, say)

    # ------------------------------------------------------------------
    # Slack API helpers
    # ------------------------------------------------------------------

    async def _download_slack_file(self, file_info: dict) -> bytes | None:
        """Download a file from Slack using the url_private_download URL."""
        url = file_info.get("url_private_download") or file_info.get("url_private", "")
        if not url:
            logger.error("No download URL in Slack file info")
            return None

        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.config.slack_bot_token}"},
                timeout=30.0,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            logger.error("Failed to download Slack file: {exc}", exc=exc)
            return None

    async def _upload_and_share_image(self, channel: str, thread_ts: str, image_path: str, caption: str | None = None):
        """Upload a local image to Slack and share it in a channel."""
        path = Path(image_path)
        if not path.exists():
            logger.error("Image file not found for Slack upload: {p}", p=image_path)
            return

        try:
            result = await self.app.client.files_upload_v2(
                channel=channel,
                file=str(path),
                filename=path.name,
                initial_comment=caption or "",
                thread_ts=thread_ts if thread_ts else None,
            )
            logger.debug("Uploaded image to Slack: {r}", r=result.get("ok"))
        except Exception as exc:
            logger.error("Failed to upload image to Slack: {exc}", exc=exc)

    async def _send_response(
        self,
        channel: str,
        thread_ts: str,
        response_text: str,
        found_images: list[dict],
        say,
    ):
        """Send agent response: images first, then text."""
        for img in found_images:
            platform = img.get("platform", "telegram")
            media_ref = img.get("media_ref") or img.get("file_id", "")
            caption = img.get("caption") or None

            if platform == "slack":
                await self._upload_and_share_image(channel, thread_ts, media_ref, caption)
            else:
                # For images from other platforms (e.g. Telegram file_ids),
                # include a note in the text response
                desc = img.get("description", "an image")
                if response_text:
                    response_text += f"\n\n_(Found image: {desc} -- originally saved via {platform})_"
                else:
                    response_text = f"_(Found image: {desc} -- originally saved via {platform})_"

        if response_text:
            await say(text=response_text[:4000], thread_ts=thread_ts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the Slack bot via Socket Mode."""
        await self.agent.start()
        await self.agent.start_background_sync(interval=60)
        self.scheduler.start()
        handler = AsyncSocketModeHandler(self.app, self.config.slack_app_token)
        await handler.start_async()

    def close(self):
        self.scheduler.close()
        self.agent.close()


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _mime_to_ext(mime_type: str) -> str:
    mapping = {
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
    return mapping.get(mime_type, "")
