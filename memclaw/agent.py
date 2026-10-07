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
import time
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from .backends import AgentBackend, build_backend
from .backends.base import ProgressCallback
from .config import MemclawConfig
from .index import MemoryIndex
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
You are a memory consolidation assistant. Your job is to distill daily memory \
logs into a curated, permanent knowledge base.

You will receive:
1. The content of several daily memory files (chronological notes, thoughts, \
saved links, voice transcriptions, etc.)
2. The current content of MEMORY.md (the permanent memory file), which may be \
empty if this is the first consolidation.

Your task:
- Extract durable facts, preferences, decisions, and important events from the \
daily files.
- Ignore transient entries: one-off reminders that have passed, trivial \
greetings, temporary notes, etc.
- Merge the extracted information with the existing MEMORY.md content. Update \
existing entries if new information supersedes them. Remove outdated entries.
- Output the complete updated MEMORY.md content in structured markdown with \
sections such as:
  ## Preferences
  ## Projects
  ## People
  ## Key Facts
  ## Decisions
  ## Important Events
- Only include sections that have content. You may add other sections if \
appropriate.
- Place the most important and frequently referenced information at the top.
- Keep the output concise — target under 5,000 characters.
- Output ONLY the markdown content for MEMORY.md. Do not include any \
explanation or preamble.
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
        # Bumped whenever consolidation rewrites MEMORY.md, so a turn can tell
        # its own MEMORY.md writes from a rewrite it hasn't seen.
        self._memory_rewrites = 0

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
        await self.backend.reset_session(key)
        self.sessions.drop(key)
        self._pending_notes.pop(key, None)
        logger.info("Started a new conversation for {k}", k=key)

    # ── Startup / sync ───────────────────────────────────────────────

    async def start(self, *, include_backend: bool = True):
        await self.index.sync()
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
                except Exception:
                    pass

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

        unconsolidated = self.store.list_unconsolidated_files(consolidated_through)
        if not unconsolidated:
            return False
        if len(unconsolidated) < self.config.consolidation_threshold and not force:
            return False

        daily_content_parts: list[str] = []
        total_chars = 0
        for path in unconsolidated:
            content = self.store.read_file(path)
            if not content.strip():
                continue
            header = f"\n### {path.stem}\n\n"
            chunk = header + content
            if total_chars + len(chunk) > 30000:
                remaining = 30000 - total_chars
                if remaining > 0:
                    daily_content_parts.append(chunk[:remaining])
                break
            daily_content_parts.append(chunk)
            total_chars += len(chunk)

        daily_text = "\n".join(daily_content_parts)
        if not daily_text.strip():
            return False

        existing_memory = self.store.read_file(self.config.memory_file)
        user_message = "## Daily Memory Files\n\n" + daily_text
        if existing_memory.strip():
            user_message += "\n\n## Current MEMORY.md\n\n" + existing_memory

        result_text = await self.backend.run_one_shot(
            system_prompt=_CONSOLIDATION_PROMPT,
            user_message=user_message,
        )

        if not result_text.strip():
            return False

        self.config.memory_file.write_text(result_text)
        self._memory_rewrites += 1

        last_date_str = unconsolidated[-1].stem
        try:
            new_consolidated_through = date.fromisoformat(last_date_str)
        except ValueError:
            new_consolidated_through = date.today()

        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        meta["consolidated_through"] = new_consolidated_through.isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))

        await self.index.index_file(self.config.memory_file)
        logger.info(
            "Consolidation complete: {n} files → MEMORY.md (through {d})",
            n=len(unconsolidated),
            d=new_consolidated_through.isoformat(),
        )
        return True

    def _schedule_consolidation(self) -> None:
        """Run the consolidation check in the background, after the reply,
        so it never delays one. At most one runs at a time."""
        if self._consolidation_task is not None and not self._consolidation_task.done():
            return
        self._consolidation_task = asyncio.create_task(self._consolidate_in_background())

    async def _consolidate_in_background(self) -> None:
        try:
            await self._maybe_consolidate()
        except Exception as exc:
            logger.exception("Background consolidation failed: {exc}", exc=exc)

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

        results = await self.search.search(message, limit=_CONTEXT_RESULTS + 2)
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
        self._found_images.clear()
        self._tools.chat_id = chat_id
        key = self._session_key(chat_id)
        generation = self._generations.get(key, 0)
        memory_rewrites = self._memory_rewrites

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
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        token_summary = (
            f"in={result.input_tokens}, out={result.output_tokens}, "
            f"cache_read={result.cache_read_tokens}, "
            f"cache_create={result.cache_creation_tokens}"
        )
        if self.backend.bills_per_token and result.cost_usd is not None:
            logger.info(
                "Agent done: {turns} turns, {ms}ms, cost ${cost:.4f} ({tokens})",
                turns=result.num_turns or 1, ms=elapsed_ms,
                cost=result.cost_usd, tokens=token_summary,
            )
        else:
            # Subscription-billed backends, or backends that don't report cost.
            logger.info(
                "Agent done: {turns} turns, {ms}ms ({tokens})",
                turns=result.num_turns or 1, ms=elapsed_ms, tokens=token_summary,
            )

        # The session has now seen these notes and these file versions. Its
        # own writes during the turn (memory_save permanent=true,
        # update_instructions) count as seen too, so they aren't sent back
        # next turn; a consolidation rewrite during the turn doesn't.
        agents_hash = _hash(_load_agent_instructions(self.config))
        if self._memory_rewrites == memory_rewrites:
            memory_hash = _hash(_load_permanent_memory(self.config))
        pending = self._pending_notes.get(key)
        if pending:
            del pending[:len(notes)]
        if result.session_id and self._generations.get(key, 0) == generation:
            self.sessions.set(key, {
                "session_id": result.session_id,
                "fingerprint": self._fingerprint,
                "agents_hash": agents_hash,
                "memory_hash": memory_hash,
            })

        self._schedule_consolidation()

        response_text = result.text or "I couldn't generate a response."
        return (response_text, list(self._found_images))

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
