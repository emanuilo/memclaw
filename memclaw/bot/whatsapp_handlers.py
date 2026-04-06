"""WhatsApp Cloud API webhook handlers for Memclaw.

Every message (text, image, voice/audio) goes through the unified MemclawAgent,
which autonomously decides whether to store, search, or just respond.

Requires a Meta Business app configured with the WhatsApp Cloud API.
Webhook verification uses the WHATSAPP_VERIFY_TOKEN.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from ..agent import MemclawAgent
from ..config import MemclawConfig
from .link_processor import LinkProcessor

# WhatsApp Cloud API base URL
_WA_API = "https://graph.facebook.com/v21.0"


class WhatsAppHandlers:
    """Routes every WhatsApp message through the unified Memclaw agent."""

    def __init__(self, config: MemclawConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.openai_client = openai_client
        self.agent = MemclawAgent(config)
        self.link_processor = LinkProcessor(openai_client)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {config.whatsapp_access_token}"},
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Webhook verification (GET)
    # ------------------------------------------------------------------

    async def verify(self, request: Request) -> PlainTextResponse:
        """Handle the WhatsApp webhook verification challenge."""
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")

        if mode == "subscribe" and token == self.config.whatsapp_verify_token:
            logger.info("WhatsApp webhook verified")
            return PlainTextResponse(challenge or "", status_code=200)

        logger.warning("WhatsApp webhook verification failed (bad token)")
        return PlainTextResponse("Forbidden", status_code=403)

    # ------------------------------------------------------------------
    # Incoming messages (POST)
    # ------------------------------------------------------------------

    async def webhook(self, request: Request) -> JSONResponse:
        """Handle incoming WhatsApp webhook events."""
        body = await request.json()

        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for message in messages:
                    sender = message.get("from", "")
                    if not self._check_sender(sender):
                        logger.warning("Ignoring message from unauthorized number: {n}", n=sender)
                        continue
                    try:
                        await self._route_message(message)
                    except Exception as exc:
                        logger.error("Error handling WhatsApp message: {exc}", exc=exc)

        return JSONResponse({"status": "ok"})

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _check_sender(self, sender: str) -> bool:
        allowed = self.config.allowed_whatsapp_numbers_list
        if not allowed:
            return True  # no allowlist = allow all
        # Normalize: strip leading '+' for comparison
        normalized = sender.lstrip("+")
        return any(a.lstrip("+") == normalized for a in allowed)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _route_message(self, message: dict):
        msg_type = message.get("type", "")
        sender = message.get("from", "")

        if msg_type == "text":
            await self._handle_text(sender, message)
        elif msg_type == "image":
            await self._handle_image(sender, message)
        elif msg_type in ("audio", "voice"):
            await self._handle_audio(sender, message)
        else:
            logger.info("Ignoring WhatsApp message type: {t}", t=msg_type)

    # ------------------------------------------------------------------
    # Text messages
    # ------------------------------------------------------------------

    async def _handle_text(self, sender: str, message: dict):
        text = message.get("text", {}).get("body", "")
        if not text:
            return
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
        await self._send_response(sender, response_text, found_images)

    # ------------------------------------------------------------------
    # Image messages
    # ------------------------------------------------------------------

    async def _handle_image(self, sender: str, message: dict):
        image_info = message.get("image", {})
        media_id = image_info.get("id", "")
        caption = image_info.get("caption", "")

        logger.info("WhatsApp image from {s}, caption={c!r}", s=sender, c=caption)

        # Download image via WhatsApp media API
        image_bytes, mime_type = await self._download_media(media_id)
        if image_bytes is None:
            await self._send_text(sender, "Sorry, I couldn't download that image.")
            return

        # Save locally
        ext = _mime_to_ext(mime_type)
        local_path = self.config.images_dir / f"{uuid.uuid4().hex}{ext}"
        local_path.write_bytes(image_bytes)

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        logger.debug("Downloaded WhatsApp image: {n} bytes -> {p}", n=len(image_bytes), p=local_path)

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

        prompt_text = f"User sent a photo. file_path={local_path}"
        if caption:
            prompt_text += f"\nCaption: {caption}"
        if link_info:
            prompt_text += link_info

        media_type = mime_type if mime_type.startswith("image/") else "image/jpeg"
        response_text, found_images = await self.agent.handle(
            prompt_text, image_b64=base64_image, image_media_type=media_type,
        )
        await self._send_response(sender, response_text, found_images)

    # ------------------------------------------------------------------
    # Audio / voice messages
    # ------------------------------------------------------------------

    async def _handle_audio(self, sender: str, message: dict):
        audio_info = message.get("audio") or message.get("voice") or {}
        media_id = audio_info.get("id", "")

        logger.info("WhatsApp voice/audio from {s}", s=sender)

        audio_bytes, mime_type = await self._download_media(media_id)
        if audio_bytes is None:
            await self._send_text(sender, "Sorry, I couldn't download that audio message.")
            return

        ext = _mime_to_ext(mime_type) or ".ogg"
        transcription = await self.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"voice{ext}", audio_bytes, mime_type or "audio/ogg"),
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
        await self._send_response(sender, response_text, found_images)

    # ------------------------------------------------------------------
    # WhatsApp Cloud API helpers
    # ------------------------------------------------------------------

    async def _download_media(self, media_id: str) -> tuple[bytes | None, str]:
        """Download media from WhatsApp Cloud API. Returns (bytes, mime_type)."""
        try:
            # Step 1: get the media URL
            resp = await self._http.get(f"{_WA_API}/{media_id}")
            resp.raise_for_status()
            media_info = resp.json()
            url = media_info.get("url", "")
            mime_type = media_info.get("mime_type", "")

            if not url:
                logger.error("No URL in media response for {id}", id=media_id)
                return None, ""

            # Step 2: download the actual file (with auth header)
            resp = await self._http.get(url)
            resp.raise_for_status()
            return resp.content, mime_type
        except Exception as exc:
            logger.error("Failed to download WhatsApp media {id}: {exc}", id=media_id, exc=exc)
            return None, ""

    async def _send_text(self, to: str, text: str):
        """Send a text message via WhatsApp Cloud API."""
        phone_id = self.config.whatsapp_phone_number_id
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text[:4096]},
        }
        try:
            resp = await self._http.post(
                f"{_WA_API}/{phone_id}/messages", json=payload,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Failed to send WhatsApp text to {to}: {exc}", to=to, exc=exc)

    async def _send_image(self, to: str, image_path: str, caption: str | None = None):
        """Upload and send a locally stored image via WhatsApp Cloud API."""
        phone_id = self.config.whatsapp_phone_number_id
        path = Path(image_path)
        if not path.exists():
            logger.error("Image file not found for WhatsApp send: {p}", p=image_path)
            return

        mime = _ext_to_mime(path.suffix)
        try:
            # Upload media
            resp = await self._http.post(
                f"{_WA_API}/{phone_id}/media",
                data={"messaging_product": "whatsapp", "type": mime},
                files={"file": (path.name, path.read_bytes(), mime)},
            )
            resp.raise_for_status()
            media_id = resp.json().get("id", "")

            if not media_id:
                logger.error("No media_id returned from WhatsApp upload")
                return

            # Send the uploaded media
            payload: dict = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "image",
                "image": {"id": media_id},
            }
            if caption:
                payload["image"]["caption"] = caption[:1024]

            resp = await self._http.post(
                f"{_WA_API}/{phone_id}/messages", json=payload,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Failed to send WhatsApp image to {to}: {exc}", to=to, exc=exc)

    async def _send_response(
        self,
        to: str,
        response_text: str,
        found_images: list[dict],
    ):
        """Send agent response: images first, then text."""
        for img in found_images:
            platform = img.get("platform", "telegram")
            media_ref = img.get("media_ref") or img.get("file_id", "")
            caption = img.get("caption") or None

            if platform == "whatsapp":
                await self._send_image(to, media_ref, caption)
            else:
                # For Telegram file_ids, we can't send them via WhatsApp.
                # Include a note in the text response instead.
                desc = img.get("description", "an image")
                if response_text:
                    response_text += f"\n\n(Found image: {desc} — originally saved via Telegram)"
                else:
                    response_text = f"(Found image: {desc} — originally saved via Telegram)"

        if response_text:
            await self._send_text(to, response_text)

    def close(self):
        self.agent.close()

    async def aclose(self):
        await self._http.aclose()
        self.agent.close()


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _mime_to_ext(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
    }
    return mapping.get(mime_type, "")


def _ext_to_mime(ext: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(ext.lower(), "image/jpeg")
