"""Every tool, in the order the LLM sees them, and how a call to one is
worded in progress messages."""

from __future__ import annotations

from typing import Any

from . import files, images, memory, reminders
from .context import Tool

TOOLS: list[Tool] = [*memory.TOOLS, *images.TOOLS, *files.TOOLS, *reminders.TOOLS]

# The JSON-schema definitions the backends hand to the LLM. The Cursor hook
# policy and the backends' allow-lists are built from these names.
TOOL_DEFINITIONS: list[dict[str, Any]] = [tool.definition for tool in TOOLS]

_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOLS}


def describe_tool_call(name: str, args: Any, limit: int = 80) -> str:
    """A short one-line description of a tool call for progress messages."""
    tool = find_tool(name)
    label, arg = (tool.step_label, tool.step_arg) if tool else (name, None)
    value = args.get(arg) if arg and isinstance(args, dict) else None
    if not isinstance(value, str) or not value.strip():
        return label
    value = " ".join(value.split())
    if len(value) > limit:
        value = value[: limit - 1] + "…"
    return f"{label}: {value}"


def find_tool(name: str) -> Tool | None:
    return _BY_NAME.get(name)
