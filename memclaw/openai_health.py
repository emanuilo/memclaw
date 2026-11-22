"""Probe the user's OpenAI key for access to every model Memclaw depends on.

Memclaw uses three OpenAI capabilities:

  - embeddings (text-embedding-3-small) — search/storage
  - chat (gpt-5-mini)                   — link summarization
  - audio.transcriptions (whisper-1)    — voice messages

A brand-new project on the OpenAI platform may have any of these disabled.
This module makes a minimal call to each one and reports which work.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from openai import AsyncOpenAI
from rich.panel import Panel
from rich.text import Text

CHAT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
WHISPER_MODEL = "whisper-1"


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


async def _probe_embedding(client: AsyncOpenAI) -> ProbeOutcome:
    try:
        await client.embeddings.create(model=EMBEDDING_MODEL, input="ok")
        return ProbeOutcome("Embeddings (search)", EMBEDDING_MODEL, True)
    except Exception as exc:
        return ProbeOutcome("Embeddings (search)", EMBEDDING_MODEL, False, _short(exc))


async def _probe_chat(client: AsyncOpenAI) -> ProbeOutcome:
    try:
        await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ok"}],
            reasoning_effort="low",
            max_completion_tokens=300,
        )
        return ProbeOutcome("Chat (link summaries)", CHAT_MODEL, True)
    except Exception as exc:
        return ProbeOutcome("Chat (link summaries)", CHAT_MODEL, False, _short(exc))


async def _probe_whisper(client: AsyncOpenAI) -> ProbeOutcome:
    """Whisper is harder to probe — a real audio file is needed.

    We send 0.5s of silence and inspect the result:
      - 200            → access confirmed.
      - 400 bad input  → access confirmed (key works, OpenAI just rejected our file).
      - Anything else  → treat as failure and surface the error.
    """
    try:
        await client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("probe.wav", _silent_wav(), "audio/wav"),
        )
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
    client = AsyncOpenAI(api_key=api_key)
    try:
        embedding = await _probe_embedding(client)
        chat = await _probe_chat(client)
        whisper = await _probe_whisper(client)
    finally:
        await client.close()
    return ProbeReport(embedding=embedding, chat=chat, whisper=whisper)


def print_probe_report(report: ProbeReport, console) -> None:
    """Render a probe report as a Rich table-ish summary."""
    body = Text()
    for outcome in report:
        marker = "[green]✓[/green]" if outcome.ok else "[red]✗[/red]"
        line = f"{marker} {outcome.name}  [dim]({outcome.model})[/dim]"
        if not outcome.ok and outcome.error:
            line += f"\n  [red]{outcome.error}[/red]"
        body.append_text(Text.from_markup(line))
        body.append("\n")

    if not report.all_ok:
        body.append("\n", style="")
        body.append_text(
            Text.from_markup(
                "[yellow]Enable the missing models for your project at[/yellow]\n"
                "[bold]https://platform.openai.com/settings/organization/limits[/bold]"
            )
        )

    console.print(
        Panel(
            body,
            title="OpenAI model access",
            border_style="green" if report.all_ok else "yellow",
        )
    )
