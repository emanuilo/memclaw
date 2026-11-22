"""What every tool works on (`ToolContext`), and the `Tool` record each
tool module lists its tools with."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from ..config import MemclawConfig
from ..memory import HybridSearch, MemoryIndex, MemoryStore
from ..reminders import ReminderScheduler


@dataclass
class ToolContext:
    """The storage the tools read and write, and the chat they run for.

    The agent builds one and updates `chat_id` before each turn; the tools
    queue the images to send with the reply in `found_images`.
    """

    config: MemclawConfig
    store: MemoryStore
    index: MemoryIndex
    search: HybridSearch
    found_images: list[dict] = field(default_factory=list)
    # The chat platform ("telegram", ...); None in the terminal.
    platform: str | None = None
    # Only the messaging bots run reminders.
    scheduler: ReminderScheduler | None = None
    chat_id: str | None = None


ToolHandler = Callable[[ToolContext, dict], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    """One tool: the definition the LLM sees, the code that runs it, and
    its wording in progress messages (`step_label`, then the `step_arg`
    argument's value if it has one)."""

    definition: dict[str, Any]
    run: ToolHandler
    step_label: str
    step_arg: str | None = None

    @property
    def name(self) -> str:
        return self.definition["name"]


async def index_written(ctx: ToolContext, path: Path) -> str:
    """Index a just-written file. The write already succeeded, so an
    indexing failure (e.g. OpenAI unreachable) is reported, not raised:
    the next index sync picks the file up."""
    try:
        await ctx.index.index_file(path)
    except Exception as exc:
        logger.warning("Indexing {p} failed, will retry on next sync: {exc}", p=path, exc=exc)
        return " (search indexing is delayed; it will be searchable soon)"
    return ""
