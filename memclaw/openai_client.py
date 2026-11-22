"""The one module that talks to OpenAI.

Memclaw uses three OpenAI capabilities:

  - embeddings (text-embedding-3-small) — search/storage
  - chat (gpt-5-mini)                   — link summarization
  - audio.transcriptions (whisper-1)    — voice messages

Everything else calls `OpenAIClient`. A brand-new project on the OpenAI
platform may have any of these models disabled, so `probe_openai` (the
`memclaw doctor` check) makes a minimal call to each one and reports which
work.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from openai import AsyncOpenAI

CHAT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
WHISPER_MODEL = "whisper-1"


class OpenAIClient:
    """The OpenAI calls Memclaw makes. Errors from the SDK are raised as is."""

    def __init__(self, api_key: str):
        self._sdk = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str], *, model: str = EMBEDDING_MODEL) -> list[list[float]]:
        """One embedding vector per text, in order."""
        response = await self._sdk.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    async def transcribe(self, audio: bytes, *, filename: str, mime_type: str) -> str:
        """Speech to text with Whisper; *filename*'s extension tells OpenAI the format."""
        result = await self._sdk.audio.transcriptions.create(
            model=WHISPER_MODEL, file=(filename, audio, mime_type),
        )
        return result.text

    async def chat(self, messages: list) -> str:
        """A short answer from the small chat model ("" if it gives none).

        *messages* are OpenAI chat messages: ``{"role": ..., "content": ...}``.
        """
        response = await self._sdk.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            reasoning_effort="low",
            max_completion_tokens=300,
        )
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        await self._sdk.close()


# ── The `memclaw doctor` probe ───────────────────────────────────────


@dataclass
class ProbeOutcome:
    name: str
    model: str
    ok: bool
    error: str | None = None


@dataclass
class ProbeReport:
    embedding: ProbeOutcome
    chat: ProbeOutcome
    whisper: ProbeOutcome

    @property
    def all_ok(self) -> bool:
        return self.embedding.ok and self.chat.ok and self.whisper.ok

    def __iter__(self):
        yield self.embedding
        yield self.chat
        yield self.whisper


def _silent_wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    """Build a minimal silent WAV (8-bit PCM, mono) for the whisper probe."""
    n = int(seconds * rate)
    data = b"\x80" * n  # 8-bit unsigned silence is centered at 128
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate, 1, 8)
    header += b"data" + struct.pack("<I", n)
    return header + data


def _short(err: BaseException, limit: int = 180) -> str:
    msg = str(err)
    return msg if len(msg) <= limit else msg[:limit] + "..."


async def _probe_embedding(client: OpenAIClient) -> ProbeOutcome:
    try:
        await client.embed(["ok"])
        return ProbeOutcome("Embeddings (search)", EMBEDDING_MODEL, True)
    except Exception as exc:
        return ProbeOutcome("Embeddings (search)", EMBEDDING_MODEL, False, _short(exc))


async def _probe_chat(client: OpenAIClient) -> ProbeOutcome:
    try:
        await client.chat([{"role": "user", "content": "ok"}])
        return ProbeOutcome("Chat (link summaries)", CHAT_MODEL, True)
    except Exception as exc:
        return ProbeOutcome("Chat (link summaries)", CHAT_MODEL, False, _short(exc))


async def _probe_whisper(client: OpenAIClient) -> ProbeOutcome:
    """Whisper is harder to probe — a real audio file is needed.

    We send 0.5s of silence and inspect the result:
      - 200            → access confirmed.
      - 400 bad input  → access confirmed (key works, OpenAI just rejected our file).
      - Anything else  → treat as failure and surface the error.
    """
    try:
        await client.transcribe(_silent_wav(), filename="probe.wav", mime_type="audio/wav")
        return ProbeOutcome("Audio (voice messages)", WHISPER_MODEL, True)
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        # A 400 on a clearly-valid request means OpenAI accepted our auth and
        # model choice but didn't like the audio itself — proves access.
        if status == 400:
            return ProbeOutcome("Audio (voice messages)", WHISPER_MODEL, True)
        return ProbeOutcome("Audio (voice messages)", WHISPER_MODEL, False, _short(exc))


async def probe_openai(api_key: str) -> ProbeReport:
    client = OpenAIClient(api_key)
    try:
        embedding = await _probe_embedding(client)
        chat = await _probe_chat(client)
        whisper = await _probe_whisper(client)
    finally:
        await client.close()
    return ProbeReport(embedding=embedding, chat=chat, whisper=whisper)
