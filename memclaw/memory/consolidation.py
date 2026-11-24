"""MEMORY.md consolidation: distill the finished days' notes into MEMORY.md.

`Consolidator` is the only code that writes MEMORY.md. It owns the prompt,
how much daily text one call gets, the dated copies in `.history/`, the
`meta.json` marker of what's done, and the background schedule (after a
reply, at most one at a time, an hour's pause after a failure).

The LLM call goes to whatever has a `run_one_shot` (the agent backend), so
this module doesn't depend on the backends.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger

from ..config import MemclawConfig
from .index import MemoryIndex
from .notes import MemoryStore
from .refs import check_references

# After a failed consolidation, wait this long before trying again rather
# than retrying after every reply.
_CONSOLIDATION_RETRY_S = 3600

# Dated copies of MEMORY.md kept from before each consolidation.
_HISTORY_DIR = ".history"
_HISTORY_KEEP = 60

# Daily-file text sent to one consolidation call (~20k tokens).
_CONSOLIDATION_BUDGET_CHARS = 80000

# A reply wrapped in a ```markdown fence despite the instructions.
_FENCED_RE = re.compile(r"^```[\w-]*\n(.*?)\n?```$", re.DOTALL)

# What the prompt's Format section requires of every MEMORY.md: it starts
# with this title and has the index heading.
_MEMORY_TITLE = "# Personal Memory"
_INDEX_RE = re.compile(r"^## Index[ \t]*$", re.MULTILINE)


_CONSOLIDATION_PROMPT = """\
You maintain MEMORY.md, the user's permanent memory. It is loaded into the \
assistant's system prompt in every conversation, so it is the assistant's \
overview of the user: a holistic profile plus an index into the detailed \
notes. The details stay where they are — the daily files \
(memory/YYYY-MM-DD.md) and the other notes are never deleted — and the \
assistant opens them when a detail is needed.

You will receive today's date, the daily files to merge in (each under a \
"### YYYY-MM-DD" header — that entry lives in memory/YYYY-MM-DD.md), a list \
of the other notes in the vault, and the current MEMORY.md, which may be \
empty. Write the complete new MEMORY.md.

PROFILE — what the assistant should know at a glance, summarized, never \
transcribed:
- Who the user is (name, age, where they live), their work, and the people \
and pets in their life and how they relate to them.
- Ongoing projects, plans and goals; health and routines; finances in broad \
strokes; lasting preferences and habits.
- Open threads worth keeping in mind (a debt, a pending decision, an \
upcoming plan), each with its date.
- Short bullets, one thing each. Leave the specifics to the notes and add a \
reference when there's more to read: "- Bekele (dog; also Izi, Beka) → \
memory/2026-04-18.md".

INDEX — where the details are, under "## Index" with subheadings such as \
Credentials, Documents, How-tos, Purchases, Notes:
- One line per thing the user may ask for later — credentials and API keys, \
document and account numbers, addresses, instructions and recipes, \
commands, receipts, measurements — naming it and pointing to the note(s) \
that hold it: "- Lion8 prod login and SSH commands → memory/2026-05-12.md, \
memory/2026-06-13.md".
- Never copy secrets, document numbers or long instructions into MEMORY.md; \
reference them. The one exception: a value that is only in the current \
MEMORY.md, with no note given to you that holds it, is kept as is.
- Reference only files you were given or that the current MEMORY.md \
already references, and only ones that hold that detail. When a detail was \
updated, point to the latest entry; when it's spread out, list each file.
- References already in the current MEMORY.md stay unless the detail was \
superseded.

Leave out entirely: moments that have passed (one-off verification codes, \
past events and deadlines, bookings for finished trips, superseded \
statuses), links and articles saved to read later (they're searchable), \
behavioural instructions for the assistant (those live in AGENTS.md), and \
small talk.

Accuracy:
- Use only what the sources say; never guess or fill gaps.
- Keep every fact with the right person: one person's birthday, document or \
preference must never end up under someone else.
- When entries conflict, the later one wins, and an explicit correction \
("ISPRAVKA", "correction", "actually") always overrides what it corrects.
- A fact in the current MEMORY.md stays unless a newer entry updates it or \
it has clearly expired.

Format:
- Markdown starting with "# Personal Memory"; profile sections (e.g. About, \
People, Work & Projects, Health, Home & Pets, Finances, Preferences, Open \
threads — only those with content), then "## Index" (always, even when \
there is nothing to index yet).
- References are written exactly as "→ memory/YYYY-MM-DD.md" (several \
separated by commas) or "→ <file>.md" for the other notes, at the end of \
the line.
- Absolute dates (YYYY-MM-DD). One language: the one the current MEMORY.md \
is written in, or English if it is empty; keep names, places and quotes as \
written.
- Fold any loose "## HH:MM - Note" entries in the current MEMORY.md into \
the profile or the index.
- Aim for about 6,000 characters in total.
- Output only the MEMORY.md content: no preamble, no code fences.
"""


class OneShotBackend(Protocol):
    """What consolidation needs from the agent backend: one tool-free call
    that returns the reply text, or raises if the call failed."""

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str: ...


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCED_RE.match(text)
    return m.group(1).strip() if m else text


def _check_memory_text(text: str) -> None:
    """Raise unless *text* has what the prompt requires of a MEMORY.md: it
    starts with the title and has the index heading.

    A backend can return error text instead of raising ("API Error: ...",
    Cursor's "I couldn't generate a response."), or a reply can come back
    empty or cut off; written over MEMORY.md, it would replace the user's
    whole profile. Raising treats it as the failed call it is: nothing is
    written and the background run waits an hour, rather than paying for a
    new call after every reply."""
    if text.startswith(_MEMORY_TITLE) and _INDEX_RE.search(text):
        return
    preview = text.strip()[:80]
    logger.warning(
        "Consolidation reply is not a MEMORY.md (no \"{t}\" title or \"## Index\"), "
        "MEMORY.md left as it was: {p!r}",
        t=_MEMORY_TITLE, p=preview,
    )
    raise RuntimeError(f"Consolidation reply is not a MEMORY.md: {preview!r}")


class Consolidator:
    """Distills the daily files not in MEMORY.md yet into it.

    `consolidate()` runs it now (`memclaw consolidate`); `schedule()` runs
    it in the background after a reply. `aclose()` / `cancel()` stop a
    background run at shutdown.
    """

    def __init__(
        self,
        config: MemclawConfig,
        store: MemoryStore,
        index: MemoryIndex,
        backend: OneShotBackend,
    ):
        self.config = config
        self._store = store
        self._index = index
        self._backend = backend
        self._task: asyncio.Task | None = None
        self._retry_at = 0.0  # monotonic time

    # ── Running ──────────────────────────────────────────────────────

    async def consolidate(self, *, force: bool = False, since: date | None = None) -> bool:
        """Distill the daily files not in MEMORY.md yet into it. Returns
        whether MEMORY.md was rewritten.

        *force* runs it below `consolidation_threshold`; *since* takes the
        daily files after that date instead of after the last one done."""
        consolidated_through = since if since is not None else self._read_consolidated_through()

        # Today's file is still being written to; it's consolidated once the
        # day is over, so nothing saved later today is skipped.
        today = date.today()
        unconsolidated = [
            path for path in self._store.list_unconsolidated_files(consolidated_through)
            if date.fromisoformat(path.stem) < today
        ]
        if not unconsolidated:
            return False
        if len(unconsolidated) < self.config.memory.consolidation_threshold and not force:
            return False

        daily_text, included = self._daily_text(unconsolidated)
        through = date.fromisoformat(included[-1].stem)
        if not daily_text.strip():
            self._write_consolidated_through(through)
            return False

        existing_memory = self._store.read_file(self.config.memory_file)
        result_text = _strip_fence(await self._backend.run_one_shot(
            system_prompt=_CONSOLIDATION_PROMPT,
            user_message=self._message(today, daily_text, existing_memory),
        ))
        refs = check_references(result_text, self.config.memory_dir)
        for ref in refs.dropped:
            logger.warning("Consolidation referenced a missing note, dropped: {r}", r=ref)
        for ref in refs.unconfirmed:
            logger.warning("Consolidation reference not confirmed by its note: {r}", r=ref)

        if not self._replace_memory(existing_memory, refs.text):
            return False
        self._write_consolidated_through(through)
        try:
            await self._index.index_file(self.config.memory_file)
        except Exception as exc:
            logger.warning("Indexing MEMORY.md after consolidation failed: {exc}", exc=exc)
        remaining = len(unconsolidated) - len(included)
        logger.info(
            "Consolidation complete: {n} files → MEMORY.md (through {d}){more}",
            n=len(included),
            d=through.isoformat(),
            more=f", {remaining} left for the next run" if remaining else "",
        )
        return True

    def schedule(self) -> None:
        """Run the consolidation check in the background, after the reply,
        so it never delays one. At most one runs at a time."""
        if self._task is not None and not self._task.done():
            return
        if time.monotonic() < self._retry_at:
            return
        self._task = asyncio.create_task(self._consolidate_in_background())

    async def _consolidate_in_background(self) -> None:
        try:
            await self.consolidate()
        except Exception as exc:
            self._retry_at = time.monotonic() + _CONSOLIDATION_RETRY_S
            logger.exception(
                "Background consolidation failed, retrying in {m} min: {exc}",
                m=_CONSOLIDATION_RETRY_S // 60, exc=exc,
            )

    def cancel(self) -> None:
        """Stop a running background consolidation without waiting for it."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def aclose(self) -> None:
        """Stop a running background consolidation and wait until it has."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ── What is sent ─────────────────────────────────────────────────

    def _daily_text(self, files: list[Path]) -> tuple[str, list[Path]]:
        """The daily files to send, and which ones that covers: whole files,
        oldest first, up to the budget; the rest wait for the next run. A
        single file over the budget is sent truncated, or it would block
        consolidation forever."""
        parts: list[str] = []
        included: list[Path] = []
        total = 0
        for path in files:
            content = self._store.read_file(path)
            part = f"\n### {path.stem}\n\n{content}" if content.strip() else ""
            if total + len(part) > _CONSOLIDATION_BUDGET_CHARS:
                if not included:
                    parts.append(part[:_CONSOLIDATION_BUDGET_CHARS])
                    included.append(path)
                break
            if part:
                parts.append(part)
                total += len(part)
            included.append(path)
        return "\n".join(parts), included

    def _message(self, today: date, daily_text: str, existing_memory: str) -> str:
        message = f"Today's date: {today.isoformat()}\n\n## Daily Memory Files\n\n{daily_text}"
        other_notes = self._other_notes_overview()
        if other_notes:
            message += f"\n\n## Other Notes\n\n{other_notes}"
        if existing_memory.strip():
            message += f"\n\n## Current MEMORY.md\n\n{existing_memory}"
        return message

    def _other_notes_overview(self) -> str:
        """The vault's notes besides MEMORY.md and the daily files — name,
        size and opening line — so MEMORY.md can reference them."""
        lines = []
        for path in self._index.indexable_files():
            if path.parent == self.config.memory_subdir or path == self.config.memory_file:
                continue
            text = self._store.read_file(path)
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
            rel = path.relative_to(self.config.memory_dir).as_posix()
            lines.append(f"- {rel} ({len(text)} chars): {first[:120]}")
        return "\n".join(lines)

    # ── Writing MEMORY.md ────────────────────────────────────────────

    def _replace_memory(self, existing_memory: str, new_memory: str) -> bool:
        """Write *new_memory* over MEMORY.md, keeping a dated copy of the old
        one. Lines someone appended by hand while consolidation was running
        are carried over; any other change meanwhile (an edit) means nothing
        is written, and consolidation retries later.

        Raises if *new_memory* isn't a MEMORY.md (an error message, a
        placeholder, an empty or cut-off reply), like a failed backend call."""
        _check_memory_text(new_memory)
        current = self._store.read_file(self.config.memory_file)
        if not current.startswith(existing_memory):
            logger.warning("MEMORY.md changed during consolidation; will retry later")
            return False
        if existing_memory.strip():
            self._save_history(existing_memory)
        added_since = current[len(existing_memory):]
        self.config.memory_file.write_text(new_memory.rstrip("\n") + "\n" + added_since)
        return True

    def _save_history(self, memory: str) -> None:
        """Keep a dated copy of MEMORY.md as it was before a consolidation,
        in a hidden directory (not indexed, not reachable by the file
        tools), pruned to the last `_HISTORY_KEEP`."""
        history = self.config.memory_dir / _HISTORY_DIR
        history.mkdir(exist_ok=True)
        (history / f"MEMORY-{datetime.now():%Y-%m-%d-%H%M%S}.md").write_text(memory)
        for old in sorted(history.glob("MEMORY-*.md"))[:-_HISTORY_KEEP]:
            old.unlink()

    # ── The marker (meta.json) ───────────────────────────────────────

    @property
    def _meta_path(self) -> Path:
        return self.config.memory_dir / "meta.json"

    def _read_meta(self) -> dict:
        try:
            meta = json.loads(self._meta_path.read_text())
        except (OSError, ValueError):  # missing or unreadable: start over
            return {}
        return meta if isinstance(meta, dict) else {}

    def _read_consolidated_through(self) -> date | None:
        """The date of the last daily file consolidated, if any."""
        try:
            return date.fromisoformat(self._read_meta().get("consolidated_through") or "")
        except ValueError:
            return None

    def _write_consolidated_through(self, through: date) -> None:
        meta = self._read_meta()
        meta["consolidated_through"] = through.isoformat()
        tmp = self._meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self._meta_path)
