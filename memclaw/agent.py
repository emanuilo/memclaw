"""Memclaw agent — raw Anthropic API implementation (no Agent SDK).

Drop-in replacement for ``agent.py``.  Uses ``anthropic.AsyncAnthropic``
with a hand-rolled agentic while-loop instead of the Claude Agent SDK.
"""

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
from .search import HybridSearch, SearchResult
from .store import MemoryStore

# ── System prompt (identical to SDK version) ─────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
Today's date: {today}

{agent_instructions}

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

# ── Tool definitions (JSON schema) ───────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "memory_save",
        "description": "Save a new memory, thought, or note",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to save"},
                "permanent": {
                    "type": "boolean",
                    "description": "If true, save to MEMORY.md instead of daily file",
                },
                "entry_type": {
                    "type": "string",
                    "description": "Type of entry: note, image, link, voice",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search through stored memories using natural language",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "image_save",
        "description": (
            "Save a local image by generating an AI description and storing it as "
            "a memory. You can see the image — describe it yourself and pass your "
            "description as content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the image file"},
                "caption": {"type": "string", "description": "Optional caption"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "telegram_image_save",
        "description": (
            "Save a Telegram image with your description for later retrieval. "
            "You MUST call this when you receive an image with a file_id. Describe "
            "the image in detail and pass the description along with the file_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed image description"},
                "file_id": {"type": "string", "description": "Telegram file_id"},
                "caption": {"type": "string", "description": "Optional caption"},
            },
            "required": ["description", "file_id"],
        },
    },
    {
        "name": "image_search",
        "description": (
            "Search for previously stored images to send to the user. "
            "Use when the user asks to retrieve, show, or send an image. "
            "The image will be sent automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for images"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_instructions",
        "description": (
            "Save a new behavioural instruction to AGENTS.md. Call this whenever "
            "the user tells you to behave a certain way, respond in a certain style, "
            "or gives any standing directive (e.g. 'always reply in Serbian', "
            "'be more concise', 'never use emojis'). Pass a short, clear rule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "The instruction to save"},
            },
            "required": ["instruction"],
        },
    },
    {
        "name": "file_write",
        "description": (
            "Create or overwrite a file inside the memory directory (~/.memclaw/). "
            "Use this when the user asks you to create a file such as todos.md, "
            "notes.md, etc. The path must be relative to ~/.memclaw/ or an "
            "absolute path under it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path (relative or absolute under ~/.memclaw/)"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "file_read",
        "description": (
            "Read a file from the memory directory (~/.memclaw/). "
            "The path must be under ~/.memclaw/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path to read"},
            },
            "required": ["file_path"],
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────

def _format_results(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        if source == "MEMORY":
            source = "permanent memory"
        parts.append(f"**[{i}]** (score: {r.score:.2f}, source: {source})\n{r.content.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "No matching memories found."


# Sonnet 4 pricing (per 1M tokens)
_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0


class MemclawAgent:
    """Unified agent for both interactive CLI and Telegram bot.

    Uses the raw Anthropic Messages API with a hand-rolled agentic loop.
    """

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self._found_images: list[dict] = []
        self._history: list[dict] = []
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    # ── Tool dispatch ────────────────────────────────────────────────

    async def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        dispatch = {
            "memory_save": self._tool_memory_save,
            "memory_search": self._tool_memory_search,
            "image_save": self._tool_image_save,
            "telegram_image_save": self._tool_telegram_image_save,
            "image_search": self._tool_image_search,
            "update_instructions": self._tool_update_instructions,
            "file_write": self._tool_file_write,
            "file_read": self._tool_file_read,
        }
        func = dispatch.get(name)
        if func is None:
            return f"Error: Unknown tool '{name}'"
        try:
            return await func(tool_input)
        except Exception as e:
            return f"Error executing {name}: {e}"

    async def _tool_memory_save(self, args: dict) -> str:
        content: str = args["content"]
        permanent: bool = args.get("permanent", False)
        entry_type: str = args.get("entry_type", "note")
        tags = args.get("tags")
        file_path = self.store.save(content, permanent=permanent, entry_type=entry_type, tags=tags)
        await self.index.index_file(file_path)
        logger.info("  → memory_save result: saved to {file}", file=file_path.name)
        return f"Memory saved to {file_path.name}"

    async def _tool_memory_search(self, args: dict) -> str:
        results = await self.search.search(args["query"], limit=args.get("limit", 10))
        formatted = _format_results(results)
        logger.info("  → memory_search result: {n} hits", n=len(results))
        for i, r in enumerate(results, 1):
            source = Path(r.file_path).stem
            snippet = r.content.strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            logger.info("    [{i}] ({score:.2f}, {src}) {snippet}", i=i, score=r.score, src=source, snippet=snippet)
        return formatted

    async def _tool_image_save(self, args: dict) -> str:
        image_path = Path(args["image_path"]).expanduser().resolve()
        caption: str = args.get("caption", "")
        if not image_path.exists():
            logger.info("  → image_save result: not found {path}", path=image_path)
            return f"Image not found: {image_path}"
        memory_content = f"**Image:** {image_path.name}\n"
        if caption:
            memory_content += f"**Caption:** {caption}\n"
        memory_content += f"**Path:** {image_path}\n"
        file_path = self.store.save(memory_content, entry_type="image")
        await self.index.index_file(file_path)
        logger.info("  → image_save result: saved {name}", name=image_path.name)
        return f"Image saved from {image_path.name}"

    async def _tool_telegram_image_save(self, args: dict) -> str:
        description: str = args["description"]
        file_id: str = args["file_id"]
        caption: str = args.get("caption", "")
        combined = f"Image: {description}"
        if caption:
            combined += f" Caption: {caption}"
        file_path = self.store.save(combined, entry_type="image")
        await self.index.index_file(file_path)
        await self.index.store_telegram_image(
            file_id=file_id, description=combined, caption=caption,
        )
        logger.info("  → telegram_image_save result: {desc}", desc=description[:100])
        return f"Image saved: {description[:100]}"

    async def _tool_image_search(self, args: dict) -> str:
        query_emb = await self.index.get_embedding(args["query"])
        candidates = self.index.search_telegram_images(query_emb, limit=5)
        if candidates:
            best_score = candidates[0]["score"]
            threshold = best_score * 0.9
            results = [r for r in candidates if r["score"] >= threshold]
        else:
            results = []
        self._found_images.extend(results)
        if results:
            lines = []
            for r in results:
                line = f"- {r['description']}"
                if r.get("caption"):
                    line += f" (caption: {r['caption']})"
                lines.append(line)
            logger.info("  → image_search result: {n} image(s) found", n=len(results))
            return (
                f"Found {len(results)} image(s):\n"
                + "\n".join(lines)
                + "\nImages will be sent automatically."
            )
        logger.info("  → image_search result: no matching images")
        return "No matching images found."

    async def _tool_update_instructions(self, args: dict) -> str:
        instruction: str = args["instruction"].strip()
        agent_file = self.config.agent_file
        entry = f"\n- {instruction}\n"
        with open(agent_file, "a") as f:
            f.write(entry)
        logger.info("  → update_instructions: appended to AGENTS.md")
        return f"Instruction saved: {instruction}"

    async def _tool_file_write(self, args: dict) -> str:
        raw = args["file_path"]
        memory_dir = self.config.memory_dir
        safe_root = memory_dir.resolve()
        p = Path(raw)
        if not p.is_absolute():
            p = memory_dir / raw
        resolved = p.expanduser().resolve()
        try:
            resolved.relative_to(safe_root)
        except ValueError:
            msg = f"Blocked: {resolved} is outside {safe_root}. Files must be under ~/.memclaw/"
            logger.warning(msg)
            return msg
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(args["content"])
        logger.info("  → file_write: wrote {path}", path=resolved)
        return f"File written: {resolved}"

    async def _tool_file_read(self, args: dict) -> str:
        raw = args["file_path"]
        memory_dir = self.config.memory_dir
        safe_root = memory_dir.resolve()
        p = Path(raw)
        if not p.is_absolute():
            p = memory_dir / raw
        resolved = p.expanduser().resolve()
        try:
            resolved.relative_to(safe_root)
        except ValueError:
            return f"Blocked: {resolved} is outside {safe_root}."
        if not resolved.exists():
            return f"File not found: {resolved}"
        text = resolved.read_text()
        logger.info("  → file_read: read {path} ({n} chars)", path=resolved, n=len(text))
        return text

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

    # ── Consolidation (unchanged — already uses raw anthropic) ───────

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
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Append full assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # If done (no tool calls), extract text and exit
            if response.stop_reason != "tool_use":
                last_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                break

            # Execute requested tool calls
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    args_str = json.dumps(block.input, ensure_ascii=False)
                    if len(args_str) > 300:
                        args_str = args_str[:300] + "..."
                    logger.info("Tool call: {name}({args})", name=block.name, args=args_str)

                    result_text = await self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})
            turn += 1

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Calculate cost (Sonnet 4 pricing)
        cost = (
            total_input_tokens * _INPUT_COST_PER_M / 1_000_000
            + total_output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        )

        logger.info(
            "Agent done: {turns} turns, {ms}ms, cost ${cost:.4f} "
            "(in={input_t}, out={output_t})",
            turns=turn + 1,
            ms=elapsed_ms,
            cost=cost,
            input_t=total_input_tokens,
            output_t=total_output_tokens,
        )

        # Append assistant response to history
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
