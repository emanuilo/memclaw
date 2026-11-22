"""Tools for the notes: save a memory, search them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..memory import SearchResult
from .context import Tool, ToolContext, index_written

_MAX_SEARCH_RESULTS = 25

MEMORY_SAVE: dict[str, Any] = {
    "name": "memory_save",
    "description": (
        "Save a new memory, thought, or note — durable facts and preferences "
        "included — to today's notes. MEMORY.md (permanent memory) is "
        "distilled from the notes automatically once a day."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The content to save"},
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
}

MEMORY_SEARCH: dict[str, Any] = {
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
}


def _format_results(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        if source == "MEMORY":
            source = "permanent memory"
        parts.append(f"**[{i}]** (score: {r.score:.2f}, source: {source})\n{r.content.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "No matching memories found."


async def memory_save(ctx: ToolContext, args: dict) -> str:
    content: str = args["content"]
    entry_type: str = args.get("entry_type", "note")
    tags = args.get("tags")
    # A "permanent" flag from an older session is ignored: MEMORY.md is
    # written only by consolidation.
    file_path = ctx.store.save(content, entry_type=entry_type, tags=tags)
    logger.info("  → memory_save result: saved to {file}", file=file_path.name)
    return f"Memory saved to {file_path.name}" + await index_written(ctx, file_path)


async def memory_search(ctx: ToolContext, args: dict) -> str:
    try:
        limit = max(1, min(int(args.get("limit") or 10), _MAX_SEARCH_RESULTS))
    except (TypeError, ValueError):
        limit = 10
    results = await ctx.search.search(args["query"], limit=limit)
    formatted = _format_results(results)
    logger.info("  → memory_search result: {n} hits", n=len(results))
    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        snippet = r.content.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        logger.info("    [{i}] ({score:.2f}, {src}) {snippet}", i=i, score=r.score, src=source, snippet=snippet)
    return formatted


TOOLS = [
    Tool(MEMORY_SAVE, memory_save, "Saving to memory"),
    Tool(MEMORY_SEARCH, memory_search, "Searching memories", "query"),
]
