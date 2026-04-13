"""Memclaw agent — raw Anthropic API with a hand-rolled agentic loop."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from .config import MemclawConfig
from .index import MemoryIndex
from .search import HybridSearch
from .store import MemoryStore
from .tools import TOOL_DEFINITIONS, ToolExecutor

# ── Prompts ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
Today's date: {today}

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

# Sonnet 4 pricing (per 1M tokens)
_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class MemclawAgent:
    """Unified agent for both interactive CLI and Telegram bot.

    Uses the raw Anthropic Messages API with a hand-rolled agentic loop.
    """

    def __init__(self, config: MemclawConfig, platform: str | None = None):
        self.config = config
        self.platform = platform
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self._found_images: list[dict] = []
        self._history: list[dict] = []
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self._tools = ToolExecutor(
            config=config,
            store=self.store,
            index=self.index,
            search=self.search,
            found_images=self._found_images,
            platform=platform,
        )

    # ── Startup / sync ───────────────────────────────────────────────

    async def start(self):
        await self.index.sync()

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

        response = await self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_CONSOLIDATION_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text

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

    # ── Main entry point (raw API agentic loop) ──────────────────────

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
    ) -> tuple[str, list[dict]]:
        self._found_images.clear()

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
            agent_instructions=agent_instructions,
            context=context,
            history=history_text,
        )

        # Build initial user message
        if image_b64:
            user_content: Any = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": message},
            ]
        else:
            user_content = message

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

        # ── Agentic loop ─────────────────────────────────────────────
        max_turns = 10
        turn = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_creation_tokens = 0
        last_text = ""
        t0 = time.perf_counter()

        while turn < max_turns:
            response = await self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
                cache_control={"type": "ephemeral"},
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            total_cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0) or 0
            total_cache_creation_tokens += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                last_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                break

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    args_str = json.dumps(block.input, ensure_ascii=False)
                    if len(args_str) > 300:
                        args_str = args_str[:300] + "..."
                    logger.info("Tool call: {name}({args})", name=block.name, args=args_str)

                    result_text = await self._tools.execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})
            turn += 1

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Cache reads are 90% cheaper than regular input tokens
        cache_read_cost = total_cache_read_tokens * _INPUT_COST_PER_M * 0.1 / 1_000_000
        cost = (
            total_input_tokens * _INPUT_COST_PER_M / 1_000_000
            + total_output_tokens * _OUTPUT_COST_PER_M / 1_000_000
            + cache_read_cost
        )

        logger.info(
            "Agent done: {turns} turns, {ms}ms, cost ${cost:.4f} "
            "(in={input_t}, out={output_t}, cache_read={cache_r}, cache_create={cache_c})",
            turns=turn + 1,
            ms=elapsed_ms,
            cost=cost,
            input_t=total_input_tokens,
            output_t=total_output_tokens,
            cache_r=total_cache_read_tokens,
            cache_c=total_cache_creation_tokens,
        )

        response_text = last_text or "I couldn't generate a response."
        self._history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
        })

        max_entries = self.config.conversation_history_limit * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        return (response_text, list(self._found_images))

    def close(self):
        task = getattr(self, "_sync_task", None)
        if task is not None and not task.done():
            task.cancel()
        self.index.close()
