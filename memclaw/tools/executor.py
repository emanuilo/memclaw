"""Runs the agent's tool calls."""

from __future__ import annotations

from typing import Any

from .catalog import find_tool
from .context import ToolContext


class ToolExecutor:
    """Executes tool calls on behalf of the agent.

    Holds the `ToolContext` (the store, index, search engine, config and
    the current chat) so that individual tools can reach them without the
    agent having to pass them on every call.
    """

    def __init__(self, context: ToolContext):
        self.context = context

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        tool = find_tool(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"
        try:
            return await tool.run(self.context, tool_input)
        except Exception as e:
            return f"Error executing {name}: {e}"
