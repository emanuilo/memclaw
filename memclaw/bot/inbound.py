"""What every chat platform does with a message, apart from the platform I/O.

A platform module receives a message, downloads its media and sends the
reply. Everything in between is here: voice to text, link summaries, the
prompt the agent gets, the agent call, and reminder delivery.

The prompt texts are part of the agent's instructions (AGENTS.md tells it
that content marked "NOT been saved yet" is for it to save or not), so they
are written once, here.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Collection

from loguru import logger

from ..agent import MemclawAgent
from ..backends.base import ProgressCallback
from ..config import MemclawConfig
from ..openai_client import OpenAIClient
from ..reminders import ReminderScheduler
from .links import LinkProcessor

TRANSCRIPTION_FAILED = (
    "I couldn't transcribe that voice message. "
    "If you set up Memclaw recently, run `memclaw doctor` to check "
    "your OpenAI key has access to the whisper-1 model."
)
NEW_CONVERSATION = "Started a new conversation."

SendText = Callable[[str, str], Awaitable[None]]  # (chat id, text)


def is_allowed(who: object, allowlist: Collection) -> bool:
    """Whether *who* may use the bot: only those on the allowlist, so an
    empty allowlist lets nobody in (on every platform that has one)."""
    return who in allowlist


def save_media(media_dir: Path, data: bytes, ext: str) -> Path:
    """Keep a received file under a platform's media dir (a random name), so
    the agent can refer to it and send it back later."""
    path = media_dir / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path


def _link_note(url: str, summary: str) -> str:
    return (
        f"\n[Link summary] {url}: {summary}"
        "\nThis summary has NOT been saved yet. Save it if the content is worth remembering."
    )


class Inbound:
    """The agent, its reminder scheduler and the message pipeline of one platform."""

    def __init__(self, config: MemclawConfig, platform: str):
        self.platform = platform
        self.openai = OpenAIClient(config.openai.api_key)
        self.links = LinkProcessor(self.openai)
        self.scheduler = ReminderScheduler(config)
        self.agent = MemclawAgent(config, platform=platform, scheduler=self.scheduler)

    # ── Prompts ──────────────────────────────────────────────────────

    async def _link_notes(self, text: str) -> list[str]:
        links = await self.links.process_links(text)
        return [_link_note(link["url"], link["summary"]) for link in links if link.get("summary")]

    async def text_prompt(self, text: str, *, quoted: str = "") -> str:
        """A text message (and the message it replies to, if any), plus a
        summary of each link in it."""
        parts = [f"[Replying to message] {quoted}"] if quoted else []
        return "\n".join([*parts, text, *await self._link_notes(text)])

    async def photo_prompt(self, media_ref: str, caption: str) -> str:
        """*media_ref* is how the agent refers to the image (image_save)."""
        prompt = f"User sent a photo. media_ref={media_ref}"
        if caption:
            prompt += f"\nCaption: {caption}" + "".join(await self._link_notes(caption))
        return prompt

    async def voice_prompt(self, transcript: str) -> str:
        # The transcription is NOT pre-saved: the agent decides.
        return (
            f"[Voice message] {transcript}"
            "\nThis transcription has NOT been saved yet. Save it if the content is worth remembering."
            + "".join(await self._link_notes(transcript))
        )

    async def transcribe(self, audio: bytes, *, filename: str, mime_type: str) -> str | None:
        """The voice message's text, or None if Whisper failed (logged; the
        platform replies with TRANSCRIPTION_FAILED)."""
        try:
            return await self.openai.transcribe(audio, filename=filename, mime_type=mime_type)
        except Exception as exc:
            logger.exception("Whisper transcription failed: {exc}", exc=exc)
            return None

    # ── The agent ────────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        *,
        chat_id: str,
        image: tuple[bytes, str] | None = None,
        on_tool: ProgressCallback | None = None,
    ) -> tuple[str, list[dict]]:
        """Run one agent turn. *image* is (bytes, media type). Returns the
        reply text and the stored images the agent chose to send back."""
        data, media_type = image or (None, "image/jpeg")
        return await self.agent.handle(
            prompt,
            image_b64=base64.b64encode(data).decode("utf-8") if data is not None else None,
            image_media_type=media_type,
            chat_id=chat_id,
            on_tool=on_tool,
        )

    async def new_conversation(self, chat_id: str) -> str:
        """The `/new` command; returns the reply."""
        await self.agent.reset_conversation(chat_id)
        return NEW_CONVERSATION

    def deliver_reminders(self, send: SendText) -> None:
        """Send this platform's due reminders with *send*; the chat's next
        turn then knows about the reminder (for a reply to it)."""
        async def deliver(chat_id: str, text: str) -> None:
            await send(chat_id, text)
            self.agent.record_reminder_fired(text, chat_id)

        self.scheduler.register_delivery(self.platform, deliver)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Index the vault, keep it in sync, start firing reminders."""
        await self.agent.start()
        await self.agent.start_background_sync(interval=60)
        self.scheduler.start()

    async def aclose(self) -> None:
        await self.agent.aclose()
        self.scheduler.close()
