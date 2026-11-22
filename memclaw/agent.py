"""Memclaw agent — backend-agnostic orchestration over a pluggable agent SDK.

The agent handles each turn: the system prompt, the per-message context and
the per-chat session bookkeeping. It wires up memory, search and MEMORY.md
consolidation (`memclaw.memory.Consolidator`), and delegates every LLM call
to an `AgentBackend` (see `memclaw.backends`). The backend is selected by
`config.agent_backend` (env var `AGENT_BACKEND`), defaulting to `claude`.

Each chat's conversation lives in the backend's own session. The system
prompt is static per session (so it can be prompt-cached); everything that
changes per turn goes into a `<context>` block at the top of the user
message instead.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from .backends import AgentBackend, build_backend
from .backends.base import ProgressCallback, TurnResult
from .config import MemclawConfig
from .index import MemoryIndex
from .instructions import (
    build_system_prompt,
    content_hash,
    instructions_fingerprint,
    load_agent_instructions,
    load_permanent_memory,
)
from .memory import Consolidator, HybridSearch, MemoryStore
from .reminders import ReminderScheduler
from .sessions import SessionStore
from .tools import ToolExecutor

# Per-turn memory search results. Kept small because every context block
# stays in the session transcript.
_CONTEXT_RESULTS = 5
_CONTEXT_SNIPPET_CHARS = 500

# Session key for the interactive terminal (no chat id).
_CLI_SESSION_KEY = "cli"


class MemclawAgent:
    """Unified agent for both interactive CLI and messaging bots.

    Turn handling and session orchestration live here; MEMORY.md
    consolidation is `self.consolidator`'s. Every LLM call is delegated to
    a pluggable `AgentBackend` chosen via `config.agent_backend`.
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
        self.consolidator = Consolidator(config, self.store, self.index, self.backend)
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
        await self.consolidator.aclose()
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

        agent_instructions = load_agent_instructions(self.config)
        permanent_memory = load_permanent_memory(self.config)
        agents_hash = content_hash(agent_instructions)
        memory_hash = content_hash(permanent_memory)

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

        t0 = time.perf_counter()
        result = await self.backend.run_turn(
            # Only used when the backend starts a new session.
            system_prompt=build_system_prompt(agent_instructions, permanent_memory),
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
        agents_hash = content_hash(load_agent_instructions(self.config))
        self._ack_notes(key, len(notes))
        if result.session_id and self._generations.get(key, 0) == generation:
            self._save_session(key, result.session_id, agents_hash, memory_hash)

        self.consolidator.schedule()

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
        task = getattr(self, "_sync_task", None)
        if task is not None and not task.done():
            task.cancel()
        self.consolidator.cancel()
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
