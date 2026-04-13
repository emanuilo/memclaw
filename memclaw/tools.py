"""Tool definitions and executor for the Memclaw agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .config import MemclawConfig
from .index import MemoryIndex
from .reminders import ReminderScheduler
from .search import HybridSearch, SearchResult
from .store import MemoryStore

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
            "Save an image with your description so it can be retrieved later. "
            "You MUST call this when you receive an image with a media_ref in "
            "the prompt, or when the user shares a local image path. Describe "
            "the image in detail and pass the media_ref verbatim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed image description"},
                "media_ref": {
                    "type": "string",
                    "description": "Opaque reference (file_id, local path, etc.) from the prompt",
                },
                "caption": {"type": "string", "description": "Optional caption"},
            },
            "required": ["description", "media_ref"],
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
    {
        "name": "reminder_create",
        "description": (
            "Schedule a reminder to be sent back to the user at a specific "
            "local time, or on a recurring interval. Use when the user says "
            "things like 'remind me tomorrow at 9am to call Alex' (one-shot) "
            "or 'remind me every 5 hours to drink water' (recurring). "
            "For recurring-only reminders without a user-specified start, "
            "you may omit fire_at and the first fire will be interval_seconds "
            "from now. Always convert the user's natural-language time to a "
            "local ISO 8601 datetime using the current local time in the system "
            "prompt as reference. Do NOT use this tool to store notes — use "
            "memory_save for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to remind the user about (short, in their own words)",
                },
                "fire_at": {
                    "type": "string",
                    "description": (
                        "Local ISO 8601 datetime for the first (or only) fire, "
                        "e.g. '2026-04-13T21:30:00'. Omit to start a recurring "
                        "reminder interval_seconds from now."
                    ),
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": (
                        "If set, the reminder recurs every N seconds after "
                        "fire_at. Convert the user's unit to seconds (e.g. "
                        "'every 5 hours' -> 18000)."
                    ),
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "reminder_list",
        "description": "List the user's pending reminders.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "reminder_cancel",
        "description": (
            "Cancel a pending reminder by its id. Call reminder_list first if "
            "the user refers to a reminder by description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Reminder id"},
            },
            "required": ["id"],
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


def _resolve_safe(raw_path: str, memory_dir: Path) -> Path | None:
    """Resolve *raw_path* and return it only if under *memory_dir*."""
    safe_root = memory_dir.resolve()
    p = Path(raw_path)
    if not p.is_absolute():
        p = memory_dir / raw_path
    resolved = p.expanduser().resolve()
    try:
        resolved.relative_to(safe_root)
    except ValueError:
        return None
    return resolved


# ── Tool executor ────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls on behalf of the agent.

    Holds references to the store, index, search engine, and config so
    that individual tool implementations can access them without the
    agent having to pass them on every call.
    """

    def __init__(
        self,
        config: MemclawConfig,
        store: MemoryStore,
        index: MemoryIndex,
        search: HybridSearch,
        found_images: list[dict],
        platform: str | None = None,
        scheduler: ReminderScheduler | None = None,
    ):
        self.config = config
        self.store = store
        self.index = index
        self.search = search
        self.found_images = found_images
        self.platform = platform
        self.scheduler = scheduler
        self.chat_id: str | None = None

        self._dispatch: dict[str, Any] = {
            "memory_save": self._memory_save,
            "memory_search": self._memory_search,
            "image_save": self._image_save,
            "image_search": self._image_search,
            "update_instructions": self._update_instructions,
            "file_write": self._file_write,
            "file_read": self._file_read,
            "reminder_create": self._reminder_create,
            "reminder_list": self._reminder_list,
            "reminder_cancel": self._reminder_cancel,
        }

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        func = self._dispatch.get(name)
        if func is None:
            return f"Error: Unknown tool '{name}'"
        try:
            return await func(tool_input)
        except Exception as e:
            return f"Error executing {name}: {e}"

    # ── Implementations ──────────────────────────────────────────────

    async def _memory_save(self, args: dict) -> str:
        content: str = args["content"]
        permanent: bool = args.get("permanent", False)
        entry_type: str = args.get("entry_type", "note")
        tags = args.get("tags")
        file_path = self.store.save(content, permanent=permanent, entry_type=entry_type, tags=tags)
        await self.index.index_file(file_path)
        logger.info("  → memory_save result: saved to {file}", file=file_path.name)
        return f"Memory saved to {file_path.name}"

    async def _memory_search(self, args: dict) -> str:
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

    async def _image_save(self, args: dict) -> str:
        description: str = args["description"]
        media_ref: str = args["media_ref"]
        caption: str = args.get("caption", "")
        platform = self.platform or "local"

        combined = f"Image: {description}"
        if caption:
            combined += f" Caption: {caption}"
        store_path = self.store.save(combined, entry_type="image")
        await self.index.index_file(store_path)
        await self.index.store_platform_image(
            platform=platform,
            media_ref=media_ref,
            description=combined,
            caption=caption,
        )
        logger.info(
            "  → image_save({platform}) result: {desc}",
            platform=platform, desc=description[:100],
        )
        return f"Image saved: {description[:100]}"

    async def _image_search(self, args: dict) -> str:
        query_emb = await self.index.get_embedding(args["query"])
        candidates = self.index.search_platform_images(query_emb, limit=5)
        if candidates:
            best_score = candidates[0]["score"]
            threshold = best_score * 0.9
            results = [r for r in candidates if r["score"] >= threshold]
        else:
            results = []
        self.found_images.extend(results)
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

    async def _update_instructions(self, args: dict) -> str:
        instruction: str = args["instruction"].strip()
        agent_file = self.config.agent_file
        entry = f"\n- {instruction}\n"
        with open(agent_file, "a") as f:
            f.write(entry)
        logger.info("  → update_instructions: appended to AGENTS.md")
        return f"Instruction saved: {instruction}"

    async def _file_write(self, args: dict) -> str:
        resolved = _resolve_safe(args["file_path"], self.config.memory_dir)
        if resolved is None:
            safe_root = self.config.memory_dir.resolve()
            msg = f"Blocked: path is outside {safe_root}. Files must be under ~/.memclaw/"
            logger.warning(msg)
            return msg
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(args["content"])
        logger.info("  → file_write: wrote {path}", path=resolved)
        return f"File written: {resolved}"

    async def _file_read(self, args: dict) -> str:
        resolved = _resolve_safe(args["file_path"], self.config.memory_dir)
        if resolved is None:
            safe_root = self.config.memory_dir.resolve()
            return f"Blocked: path is outside {safe_root}."
        if not resolved.exists():
            return f"File not found: {resolved}"
        text = resolved.read_text()
        logger.info("  → file_read: read {path} ({n} chars)", path=resolved, n=len(text))
        return text

    # ── Reminders ────────────────────────────────────────────────────

    def _require_reminder_ctx(self) -> str | None:
        if self.scheduler is None:
            return "Reminders are only available when running in a messaging bot (telegram/whatsapp/slack)."
        if not self.platform or not self.chat_id:
            return "Reminders require a messaging context (missing platform/chat_id)."
        return None

    async def _reminder_create(self, args: dict) -> str:
        err = self._require_reminder_ctx()
        if err:
            return err

        text: str = args["text"].strip()
        interval: int | None = args.get("interval_seconds")
        fire_at_str: str | None = args.get("fire_at")

        if fire_at_str:
            try:
                fire_at = datetime.fromisoformat(fire_at_str)
            except ValueError:
                return f"Invalid fire_at: {fire_at_str!r}. Use local ISO 8601."
            if fire_at.tzinfo is not None:
                fire_at = fire_at.astimezone().replace(tzinfo=None)
        elif interval:
            from datetime import timedelta
            fire_at = datetime.now() + timedelta(seconds=interval)
        else:
            return "Provide either fire_at or interval_seconds (or both)."

        if interval is not None and interval < 60:
            return "interval_seconds must be at least 60."

        rid = self.scheduler.create(  # type: ignore[union-attr]
            platform=self.platform,  # type: ignore[arg-type]
            chat_id=self.chat_id,  # type: ignore[arg-type]
            text=text,
            fire_at=fire_at,
            interval_seconds=interval,
        )
        logger.info(
            "  → reminder_create: id={id} fire_at={f} every={i}s",
            id=rid, f=fire_at.isoformat(), i=interval,
        )
        if interval:
            return (
                f"Reminder #{rid} scheduled for {fire_at.isoformat(timespec='minutes')} "
                f"and every {interval} seconds after."
            )
        return f"Reminder #{rid} scheduled for {fire_at.isoformat(timespec='minutes')}."

    async def _reminder_list(self, args: dict) -> str:
        err = self._require_reminder_ctx()
        if err:
            return err
        items = self.scheduler.list_for(self.platform, self.chat_id)  # type: ignore[union-attr,arg-type]
        if not items:
            return "No pending reminders."
        lines = []
        for it in items:
            line = f"#{it['id']} at {it['fire_at']}: {it['text']}"
            if it["interval_seconds"]:
                line += f" (every {it['interval_seconds']}s)"
            lines.append(line)
        return "\n".join(lines)

    async def _reminder_cancel(self, args: dict) -> str:
        err = self._require_reminder_ctx()
        if err:
            return err
        rid = int(args["id"])
        ok = self.scheduler.cancel(  # type: ignore[union-attr]
            rid, platform=self.platform, chat_id=self.chat_id,  # type: ignore[arg-type]
        )
        return f"Reminder #{rid} cancelled." if ok else f"No pending reminder #{rid}."
