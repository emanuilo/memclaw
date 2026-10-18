"""Memclaw agent — backend-agnostic orchestration over a pluggable agent SDK.

The agent owns memory, search, consolidation, the system-prompt shape and
the per-chat session bookkeeping, but delegates every LLM call to an
`AgentBackend` (see `memclaw.backends`). The backend is selected by
`config.agent_backend` (env var `AGENT_BACKEND`), defaulting to `claude`.

Each chat's conversation lives in the backend's own session. The system
prompt is static per session (so it can be prompt-cached); everything that
changes per turn goes into a `<context>` block at the top of the user
message instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from .backends import AgentBackend, build_backend
from .backends.base import ProgressCallback, TurnResult
from .config import MemclawConfig
from .index import MemoryIndex
from .memory_refs import check_references
from .reminders import ReminderScheduler
from .search import HybridSearch
from .sessions import SessionStore
from .store import MemoryStore
from .tools import ToolExecutor

# ── Prompts ──────────────────────────────────────────────────────────

# Static for the whole session. Changing this template changes the session
# fingerprint, so existing sessions start fresh instead of being resumed.
_SYSTEM_PROMPT_TEMPLATE = """\
{agent_instructions}

=== REPLY FORMATTING ===
Replies are delivered to messaging apps with limited markdown support. Use \
ONLY this minimal syntax — anything else leaks as literal characters:
- Bold: `*bold*` (single asterisk). NEVER use `**double asterisks**`.
- Italic: `_italic_`.
- Bullet lists: plain `- item` on its own line.
- Paragraphs: separate with a blank line.
- Headings (`#`, `##`, ...) are NOT supported — use a bold line on its own \
(e.g. `*Section name*`) followed by a blank line instead.
- No backticks or fenced code blocks.
- No `[label](url)` links — write the bare URL.

=== PER-MESSAGE CONTEXT ===
Each user message starts with a <context> block added by Memclaw, not typed \
by the user. It holds the current local time (use it for dates and \
reminders), memories that may be relevant to that message, reminders that \
were delivered to the user since their last message, and the new content of \
AGENTS.md or MEMORY.md when either file changes — such an update supersedes \
the version in this system prompt and any earlier update. Don't mention the \
block itself to the user.

=== PERMANENT MEMORY (MEMORY.md) ===
An overview of the user. Lines ending in "→ memory/2026-04-18.md" (or \
another file) point to the notes holding the details: when the user needs \
a detail — an exact value, a number, instructions — open the note with \
file_read and answer from it, rather than from the summary. memory_search \
finds anything else.

{permanent_memory}

IMPORTANT: When the user gives you a behavioural instruction (e.g. "always respond \
in Spanish", "be more formal", "never use emojis"), you MUST call the \
update_instructions tool to save it. These are rules you should follow in every \
future conversation.
"""

# MEMORY.md is meant to stay under ~5,000 characters (see consolidation);
# this cap only guards against a runaway file.
_MEMORY_PROMPT_CHARS = 20000

# Per-turn memory search results. Kept small because every context block
# stays in the session transcript.
_CONTEXT_RESULTS = 5
_CONTEXT_SNIPPET_CHARS = 500

# Session key for the interactive terminal (no chat id).
_CLI_SESSION_KEY = "cli"

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


def _load_agent_instructions(config: MemclawConfig) -> str:
    agent_file = config.agent_file
    if agent_file.exists():
        return agent_file.read_text().strip()
    return "You are Memclaw, a personal memory assistant."


def _load_permanent_memory(config: MemclawConfig) -> str:
    memory = config.memory_file.read_text().strip() if config.memory_file.exists() else ""
    if not memory:
        return "(empty)"
    if len(memory) > _MEMORY_PROMPT_CHARS:
        return memory[:_MEMORY_PROMPT_CHARS] + "\n\n[... truncated; use memory_search for more]"
    return memory


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCED_RE.match(text)
    return m.group(1).strip() if m else text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def instructions_fingerprint(backend_name: str) -> str:
    """Hash of the code-level instructions a session is started with.

    Claude Code keeps a session's original system prompt when it is resumed,
    so a session started under a different template (or backend) must not be
    resumed. AGENTS.md / MEMORY.md are tracked separately per session and
    sent as updates instead, so editing them doesn't reset conversations.
    """
    return _hash(f"{backend_name}\0{_SYSTEM_PROMPT_TEMPLATE}")


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
threads — only those with content), then "## Index".
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


class MemclawAgent:
    """Unified agent for both interactive CLI and messaging bots.

    Memory, search, consolidation, and session orchestration live here.
    Every LLM call is delegated to a pluggable `AgentBackend` chosen via
    `config.agent_backend`.
    """

    def __init__(
        self,
        config: MemclawConfig,
        platform: str | None = None,
        *,
        scheduler: ReminderScheduler | None = None,
        backend: AgentBackend | None = None,
    ):
        self.config = config
        self.platform = platform
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self.scheduler = scheduler
        self._found_images: list[dict] = []
        self._tools = ToolExecutor(
            config=config,
            store=self.store,
            index=self.index,
            search=self.search,
            found_images=self._found_images,
            platform=platform,
            scheduler=scheduler,
        )
        self.backend: AgentBackend = backend or build_backend(config)
        self._backend_started = False
        self.sessions = SessionStore(config.memory_dir / "sessions.json")
        self._fingerprint = instructions_fingerprint(self.backend.name)
        # Delivered reminders not yet seen by the chat's session.
        self._pending_notes: dict[str, list[str]] = {}
        # Bumped by /new so a turn that was running doesn't re-save the
        # session it just reset.
        self._generations: dict[str, int] = {}
        self._consolidation_task: asyncio.Task | None = None
        self._consolidation_retry_at = 0.0  # monotonic time
        # The tool executor holds per-turn state (chat id, found images), so
        # turns run one at a time, even across chats.
        self._turn_lock = asyncio.Lock()

    # ── Sessions ─────────────────────────────────────────────────────

    def _session_key(self, chat_id: str | None) -> str:
        if chat_id is None:
            return _CLI_SESSION_KEY
        return f"{self.platform}:{chat_id}" if self.platform else str(chat_id)

    def record_reminder_fired(self, text: str, chat_id: str | None = None):
        """Remember a delivered reminder so the chat's next message carries
        it, giving the agent context if the user replies to it."""
        self._pending_notes.setdefault(self._session_key(chat_id), []).append(text)

    async def reset_conversation(self, chat_id: str | None = None) -> None:
        """Start a fresh conversation for *chat_id* (the `/new` command)."""
        key = self._session_key(chat_id)
        self._generations[key] = self._generations.get(key, 0) + 1
        # Forget the session before awaiting the backend: a message sent
        # right after /new may run while the reset is still disconnecting,
        # and must not resume the old session.
        self.sessions.drop(key)
        self._pending_notes.pop(key, None)
        await self.backend.reset_session(key)
        logger.info("Started a new conversation for {k}", k=key)

    # ── Startup / sync ───────────────────────────────────────────────

    async def start(self, *, include_backend: bool = True):
        try:
            await self.index.sync()
        except Exception as exc:
            # E.g. OpenAI unreachable. The bot still works; the background
            # sync (or the next start) indexes what's missing.
            logger.warning("Index sync at startup failed: {exc}", exc=exc)
        if include_backend:
            await self.backend.on_agent_start(self._tools)
            self._backend_started = True

    async def aclose(self):
        """Async shutdown: stop consolidation, release backend resources,
        then sync cleanup."""
        task = self._consolidation_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._backend_started:
            await self.backend.on_agent_shutdown()
            self._backend_started = False
        self._close_sync_only()

    async def start_background_sync(self, interval: int = 60):
        index = self.index

        async def _sync_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await index.sync()
                except Exception as exc:
                    logger.warning("Background index sync failed: {exc}", exc=exc)

        self._sync_task = asyncio.create_task(_sync_loop())

    # ── Consolidation ────────────────────────────────────────────────

    async def _maybe_consolidate(
        self,
        *,
        force: bool = False,
        consolidated_through_override: date | None = None,
    ) -> bool:
        meta_path = self.config.memory_dir / "meta.json"

        consolidated_through: date | None = None
        if consolidated_through_override is not None:
            consolidated_through = consolidated_through_override
        elif meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                ct = meta.get("consolidated_through")
                if ct:
                    consolidated_through = date.fromisoformat(ct)
            except (json.JSONDecodeError, ValueError):
                pass

        # Today's file is still being written to; it's consolidated once the
        # day is over, so nothing saved later today is skipped.
        today = date.today()
        unconsolidated = [
            path for path in self.store.list_unconsolidated_files(consolidated_through)
            if date.fromisoformat(path.stem) < today
        ]
        if not unconsolidated:
            return False
        if len(unconsolidated) < self.config.consolidation_threshold and not force:
            return False

        # Whole files only, oldest first, up to the budget; the rest wait for
        # the next run. A single file over the budget is sent truncated, or
        # it would block consolidation forever.
        daily_content_parts: list[str] = []
        included: list[Path] = []
        total_chars = 0
        for path in unconsolidated:
            content = self.store.read_file(path)
            chunk = f"\n### {path.stem}\n\n{content}" if content.strip() else ""
            if total_chars + len(chunk) > _CONSOLIDATION_BUDGET_CHARS:
                if not included:
                    daily_content_parts.append(chunk[:_CONSOLIDATION_BUDGET_CHARS])
                    included.append(path)
                break
            if chunk:
                daily_content_parts.append(chunk)
                total_chars += len(chunk)
            included.append(path)

        daily_text = "\n".join(daily_content_parts)
        new_consolidated_through = date.fromisoformat(included[-1].stem)
        if not daily_text.strip():
            self._write_consolidated_through(meta_path, new_consolidated_through)
            return False

        existing_memory = self.store.read_file(self.config.memory_file)
        user_message = (
            f"Today's date: {today.isoformat()}\n\n## Daily Memory Files\n\n" + daily_text
        )
        other_notes = self._other_notes_overview()
        if other_notes:
            user_message += "\n\n## Other Notes\n\n" + other_notes
        if existing_memory.strip():
            user_message += "\n\n## Current MEMORY.md\n\n" + existing_memory

        result_text = await self.backend.run_one_shot(
            system_prompt=_CONSOLIDATION_PROMPT,
            user_message=user_message,
        )
        result_text = _strip_fence(result_text)
        if not result_text:
            return False
        refs = check_references(result_text, self.config.memory_dir)
        result_text = refs.text
        for ref in refs.dropped:
            logger.warning("Consolidation referenced a missing note, dropped: {r}", r=ref)
        for ref in refs.unconfirmed:
            logger.warning("Consolidation reference not confirmed by its note: {r}", r=ref)

        # Someone may have appended to MEMORY.md by hand while the backend
        # was working: carry that over instead of overwriting it. Any other
        # change (an edit) — try again later.
        current_memory = self.store.read_file(self.config.memory_file)
        if not current_memory.startswith(existing_memory):
            logger.warning("MEMORY.md changed during consolidation; will retry later")
            return False
        added_since = current_memory[len(existing_memory):]

        if existing_memory.strip():
            self._save_history(existing_memory)
        self.config.memory_file.write_text(result_text.rstrip("\n") + "\n" + added_since)
        self._write_consolidated_through(meta_path, new_consolidated_through)

        try:
            await self.index.index_file(self.config.memory_file)
        except Exception as exc:
            logger.warning("Indexing MEMORY.md after consolidation failed: {exc}", exc=exc)
        remaining = len(unconsolidated) - len(included)
        logger.info(
            "Consolidation complete: {n} files → MEMORY.md (through {d}){more}",
            n=len(included),
            d=new_consolidated_through.isoformat(),
            more=f", {remaining} left for the next run" if remaining else "",
        )
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

    def _other_notes_overview(self) -> str:
        """The vault's notes besides MEMORY.md and the daily files — name,
        size and opening line — so MEMORY.md can reference them."""
        lines = []
        for path in self.index.indexable_files():
            if path.parent == self.config.memory_subdir or path == self.config.memory_file:
                continue
            text = self.store.read_file(path)
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
            rel = path.relative_to(self.config.memory_dir).as_posix()
            lines.append(f"- {rel} ({len(text)} chars): {first[:120]}")
        return "\n".join(lines)

    @staticmethod
    def _write_consolidated_through(meta_path: Path, through: date) -> None:
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        meta["consolidated_through"] = through.isoformat()
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(meta_path)

    def _schedule_consolidation(self) -> None:
        """Run the consolidation check in the background, after the reply,
        so it never delays one. At most one runs at a time."""
        if self._consolidation_task is not None and not self._consolidation_task.done():
            return
        if time.monotonic() < self._consolidation_retry_at:
            return
        self._consolidation_task = asyncio.create_task(self._consolidate_in_background())

    async def _consolidate_in_background(self) -> None:
        try:
            await self._maybe_consolidate()
        except Exception as exc:
            self._consolidation_retry_at = time.monotonic() + _CONSOLIDATION_RETRY_S
            logger.exception(
                "Background consolidation failed, retrying in {m} min: {exc}",
                m=_CONSOLIDATION_RETRY_S // 60, exc=exc,
            )

    # ── Context builder ──────────────────────────────────────────────

    async def build_context(
        self,
        message: str,
        *,
        notes: list[str] | tuple[str, ...] = (),
        agents_update: str | None = None,
        memory_update: str | None = None,
    ) -> str:
        """The per-turn `<context>` block prefixed to the user message.

        It stays in the session transcript, so it only carries what changes
        per turn and keeps memory snippets short. MEMORY.md is already in
        the system prompt, so its search hits are skipped.
        """
        now = datetime.now()
        parts = [
            "<context>",
            f"Current local time: {now:%Y-%m-%d %H:%M} ({now:%A})",
        ]
        if notes:
            parts.append("Reminders delivered since the user's last message:")
            parts.extend(f"- {note}" for note in notes)
        if agents_update is not None:
            parts.append("Updated AGENTS.md (supersedes the earlier version):")
            parts.append(agents_update)
        if memory_update is not None:
            parts.append("Updated permanent memory, MEMORY.md (supersedes the earlier version):")
            parts.append(memory_update)

        try:
            results = await self.search.search(message, limit=_CONTEXT_RESULTS + 2)
        except Exception as exc:
            # E.g. OpenAI unreachable: answer without memories rather than
            # failing the message. memory_search reports its own errors.
            logger.warning("Context memory search failed: {exc}", exc=exc)
            results = []
        results = [
            r for r in results if Path(r.file_path).name != self.config.memory_file.name
        ][:_CONTEXT_RESULTS]
        if results:
            parts.append("Relevant memories:")
            for r in results:
                snippet = " ".join(r.content.split())
                if len(snippet) > _CONTEXT_SNIPPET_CHARS:
                    snippet = snippet[:_CONTEXT_SNIPPET_CHARS] + "…"
                parts.append(f"- [{Path(r.file_path).stem}] {snippet}")

        parts.append("</context>")
        return "\n".join(parts)

    # ── Main entry point ─────────────────────────────────────────────

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        chat_id: str | None = None,
        on_tool: ProgressCallback | None = None,
    ) -> tuple[str, list[dict]]:
        async with self._turn_lock:
            return await self._handle(
                message,
                image_b64=image_b64,
                image_media_type=image_media_type,
                chat_id=chat_id,
                on_tool=on_tool,
            )

    async def _handle(
        self,
        message: str,
        *,
        image_b64: str | None,
        image_media_type: str,
        chat_id: str | None,
        on_tool: ProgressCallback | None,
    ) -> tuple[str, list[dict]]:
        self._found_images.clear()
        self._tools.chat_id = chat_id
        key = self._session_key(chat_id)
        generation = self._generations.get(key, 0)

        agent_instructions = _load_agent_instructions(self.config)
        permanent_memory = _load_permanent_memory(self.config)
        agents_hash = _hash(agent_instructions)
        memory_hash = _hash(permanent_memory)

        # A resumed (or live) session has the AGENTS.md / MEMORY.md it last
        # saw; send whichever changed since, once.
        entry = self.sessions.get(key, self._fingerprint)
        notes = list(self._pending_notes.get(key, ()))
        context = await self.build_context(
            message,
            notes=notes,
            agents_update=(
                agent_instructions
                if entry and entry.get("agents_hash") != agents_hash else None
            ),
            memory_update=(
                permanent_memory
                if entry and entry.get("memory_hash") != memory_hash else None
            ),
        )

        # Only used when the backend starts a new session.
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            agent_instructions=agent_instructions,
            permanent_memory=permanent_memory,
        )

        t0 = time.perf_counter()
        result = await self.backend.run_turn(
            system_prompt=system_prompt,
            context=context,
            user_message=message,
            tool_executor=self._tools,
            session_key=key,
            resume_session_id=entry["session_id"] if entry else None,
            image_b64=image_b64,
            image_media_type=image_media_type,
            max_turns=10,
            on_tool=on_tool,
        )
        self._log_turn(result, int((time.perf_counter() - t0) * 1000))

        # The session has now seen these notes and these file versions. Its
        # own AGENTS.md writes during the turn (update_instructions) count as
        # seen too, so they aren't sent back next turn. It never writes
        # MEMORY.md, so a change to it during the turn (a consolidation) is
        # still unseen and goes into the next message.
        agents_hash = _hash(_load_agent_instructions(self.config))
        self._ack_notes(key, len(notes))
        if result.session_id and self._generations.get(key, 0) == generation:
            self._save_session(key, result.session_id, agents_hash, memory_hash)

        self._schedule_consolidation()

        response_text = result.text or "I couldn't generate a response."
        return (response_text, list(self._found_images))

    def _log_turn(self, result: TurnResult, elapsed_ms: int) -> None:
        tokens = (
            f"in={result.input_tokens}, out={result.output_tokens}, "
            f"cache_read={result.cache_read_tokens}, "
            f"cache_create={result.cache_creation_tokens}"
        )
        turns = result.num_turns or 1
        if self.backend.bills_per_token and result.cost_usd is not None:
            logger.info(
                "Agent done: {turns} turns, {ms}ms, cost ${cost:.4f} ({tokens})",
                turns=turns, ms=elapsed_ms, cost=result.cost_usd, tokens=tokens,
            )
        else:
            # Subscription-billed backends, or backends that don't report cost.
            logger.info(
                "Agent done: {turns} turns, {ms}ms ({tokens})",
                turns=turns, ms=elapsed_ms, tokens=tokens,
            )

    def _ack_notes(self, key: str, seen: int) -> None:
        """Drop the first *seen* pending notes, the ones this turn carried.
        Only after the turn succeeded, and only those: a reminder delivered
        during the turn stays queued for the next message."""
        pending = self._pending_notes.get(key)
        if pending:
            del pending[:seen]

    def _save_session(self, key: str, session_id: str, agents_hash: str, memory_hash: str) -> None:
        self.sessions.set(key, {
            "session_id": session_id,
            "fingerprint": self._fingerprint,
            "agents_hash": agents_hash,
            "memory_hash": memory_hash,
        })

    def _close_sync_only(self) -> None:
        for task in (getattr(self, "_sync_task", None), self._consolidation_task):
            if task is not None and not task.done():
                task.cancel()
        self.index.close()

    def close(self):
        """Sync cleanup including backend teardown when no event loop is running.

        When an asyncio loop is already running in this thread, this cannot
        await :meth:`AgentBackend.on_agent_shutdown`; use ``await aclose()``
        instead for full teardown (e.g. stopping the Cursor MCP HTTP server).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.backend.on_agent_shutdown())
        self._close_sync_only()
