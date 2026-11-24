"""Tools for files in the memory directory: AGENTS.md instructions, and
reading and writing the user's own files (never Memclaw's internal ones)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from .context import Tool, ToolContext, index_written

# Memclaw's own state inside the memory dir. The agent may not read or
# write it: the database, the session store and the consolidation marker
# would be corrupted by a write, .env holds the API keys, and the WhatsApp /
# Slack dirs hold the pairing session and downloaded media. Hidden
# directories (e.g. .cursor/, holding the Cursor tool-restriction hooks) are
# covered by the dot-prefix rule.
_INTERNAL_NAMES = frozenset({"sessions.json", "sessions.tmp", "meta.json"})
_INTERNAL_DIRS = frozenset({"whatsapp", "slack"})

UPDATE_INSTRUCTIONS: dict[str, Any] = {
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
}

FILE_WRITE: dict[str, Any] = {
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
}

FILE_READ: dict[str, Any] = {
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
}


def _resolve_safe(raw_path: str, memory_dir: Path) -> Path | None:
    """Resolve *raw_path* and return it only if under *memory_dir* and not
    one of Memclaw's internal files."""
    safe_root = memory_dir.resolve()
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = memory_dir / p
    resolved = p.resolve()
    try:
        rel = resolved.relative_to(safe_root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    if any(part.startswith(".") for part in parts):
        return None
    if parts[0] in _INTERNAL_DIRS or rel.name in _INTERNAL_NAMES:
        return None
    if rel.name.startswith("memclaw.db") or rel.suffix == ".log":
        return None
    return resolved


async def update_instructions(ctx: ToolContext, args: dict) -> str:
    instruction: str = args["instruction"].strip()
    agent_file = ctx.config.agent_file
    entry = f"\n- {instruction}\n"
    with open(agent_file, "a") as f:
        f.write(entry)
    logger.info("  → update_instructions: appended to AGENTS.md")
    return f"Instruction saved: {instruction}"


async def file_write(ctx: ToolContext, args: dict) -> str:
    config = ctx.config
    resolved = _resolve_safe(args["file_path"], config.memory_dir)
    if resolved is None:
        safe_root = config.memory_dir.resolve()
        msg = (
            f"Blocked: path is outside {safe_root} or is one of Memclaw's "
            "internal files. Files must be under ~/.memclaw/"
        )
        logger.warning(msg)
        return msg
    rel = resolved.relative_to(config.memory_dir.resolve())
    if rel == Path(config.memory_file.name) or rel.parts[0] == config.memory_subdir.name:
        # MEMORY.md is written only by consolidation, and the daily notes
        # are the append-only record it is rebuilt from.
        return (
            "Blocked: MEMORY.md and the daily notes in memory/ are maintained "
            "by Memclaw. Use memory_save to add a memory."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(args["content"])
    logger.info("  → file_write: wrote {path}", path=resolved)
    # Index under the same (unresolved) path the index sync uses.
    path = config.memory_dir / rel
    note = await index_written(ctx, path) if ctx.index.is_indexable(path) else ""
    return f"File written: {resolved}{note}"


async def file_read(ctx: ToolContext, args: dict) -> str:
    resolved = _resolve_safe(args["file_path"], ctx.config.memory_dir)
    if resolved is None:
        safe_root = ctx.config.memory_dir.resolve()
        return f"Blocked: path is outside {safe_root} or is one of Memclaw's internal files."
    if not resolved.exists():
        return f"File not found: {resolved}"
    text = resolved.read_text()
    logger.info("  → file_read: read {path} ({n} chars)", path=resolved, n=len(text))
    return text


TOOLS = [
    Tool(UPDATE_INSTRUCTIONS, update_instructions, "Updating my instructions"),
    Tool(FILE_WRITE, file_write, "Writing", "file_path"),
    Tool(FILE_READ, file_read, "Reading", "file_path"),
]
