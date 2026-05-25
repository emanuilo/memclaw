"""Memclaw agent — backend-agnostic orchestration over a pluggable agent SDK.

The agent owns memory, search, history, consolidation, and the system-prompt
shape, but delegates every LLM call to an `AgentBackend` (see
`memclaw.backends`). The backend is selected by `config.agent_backend`
(env var `AGENT_BACKEND`), defaulting to `claude`.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from .backends import AgentBackend, build_backend
from .config import MemclawConfig
from .index import MemoryIndex
from .reminders import ReminderScheduler
from .search import HybridSearch
from .store import MemoryStore
from .tools import ToolExecutor

# ── Prompts ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
Today's date: {today}
Current local time: {now}

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

=== MEMORY CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{history}

IMPORTANT: When the user gives you a behavioural instruction (e.g. "always respond \
in Spanish", "be more formal", "never use emojis"), you MUST call the \
update_instructions tool to save it. These are rules you should follow in every \
future conversation.
"""


def _load_agent_instructions(config: MemclawConfig) -> str:
    agent_file = config.agent_file
    if agent_file.exists():
        return agent_file.read_text().strip()
    return "You are Memclaw, a personal memory assistant."


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

    Memory, search, history, and consolidation orchestration live here.
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
        self._history: list[dict] = []
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

    # ── Startup / sync ───────────────────────────────────────────────

    def record_reminder_fired(self, text: str):
        """Append a delivered reminder to history so the agent has context
        if the user replies to it."""
        self._history.append({
            "role": "assistant",
            "content": f"[Reminder fired] {text}",
            "timestamp": datetime.now().isoformat(),
        })
        max_entries = self.config.conversation_history_limit * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

    async def start(self, *, include_backend: bool = True):
        await self.index.sync()
        if include_backend:
            await self.backend.on_agent_start(self._tools)
            self._backend_started = True

    async def aclose(self):
        """Async shutdown: release backend resources then sync cleanup."""
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

    # ── Context builder ──────────────────────────────────────────────

    async def build_context(self, message: str) -> str:
        parts: list[str] = []

        memory_content = self.store.read_file(self.config.memory_file)
        if memory_content.strip():
            parts.append("### Permanent Memory")
            if len(memory_content) <= 4000:
                parts.append(memory_content)
            else:
                parts.append(memory_content[:2000])
                memory_results = await self.search.search(
                    message, limit=3, file_filter="MEMORY.md"
                )
                if memory_results:
                    parts.append("\n#### Relevant Permanent Memory Sections")
                    for r in memory_results:
                        parts.append(r.content.strip())

        results = await self.search.search(message, limit=10)
        if results:
            parts.append("\n### Relevant Memories")
            for r in results:
                source = Path(r.file_path).stem
                parts.append(f"[{source}] {r.content.strip()}")

        return "\n\n".join(parts) if parts else "No memories found yet."

    # ── Main entry point ─────────────────────────────────────────────

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        chat_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        self._found_images.clear()
        self._tools.chat_id = chat_id

        try:
            await self._maybe_consolidate()
        except Exception as exc:
            logger.warning("Consolidation check failed: {exc}", exc=exc)

        history_content = "[User sent a photo]" if image_b64 else message
        self._history.append({
            "role": "user",
            "content": history_content,
            "timestamp": datetime.now().isoformat(),
        })

        context = await self.build_context(message)

        history_snapshot = self._history[:-1]
        if history_snapshot:
            history_lines = []
            for entry in history_snapshot:
                role = "User" if entry["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {entry['content']}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "(no prior messages)"

        agent_instructions = _load_agent_instructions(self.config)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            today=date.today().isoformat(),
            now=datetime.now().replace(microsecond=0).isoformat(),
            agent_instructions=agent_instructions,
            context=context,
            history=history_text,
        )

        t0 = time.perf_counter()
        result = await self.backend.run_turn(
            system_prompt=system_prompt,
            user_message=message,
            tool_executor=self._tools,
            image_b64=image_b64,
            image_media_type=image_media_type,
            max_turns=10,
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

        response_text = result.text or "I couldn't generate a response."
        self._history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
        })

        max_entries = self.config.conversation_history_limit * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        return (response_text, list(self._found_images))

    def _close_sync_only(self) -> None:
        task = getattr(self, "_sync_task", None)
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
