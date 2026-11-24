"""Shared MCP server builder for Memclaw tool execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from ..tools import TOOL_DEFINITIONS

if TYPE_CHECKING:
    from ..tools import ToolExecutor

MCP_SERVER_NAME = "memclaw"


def build_memclaw_mcp_server(executor: "ToolExecutor") -> Server:
    """Build an in-process MCP server that routes tool calls to *executor*."""
    server = Server(MCP_SERVER_NAME, version="1.0.0")
    tool_names = {defn["name"] for defn in TOOL_DEFINITIONS}
    cached_tools = [
        Tool(
            name=defn["name"],
            description=defn["description"],
            inputSchema=defn["input_schema"],
        )
        for defn in TOOL_DEFINITIONS
    ]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return cached_tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        if name not in tool_names:
            raise ValueError(f"Tool {name!r} not found")

        logger.info("MCP call: {name}({args})", name=name, args=arguments)
        text = await executor.execute(name, arguments or {})
        return CallToolResult(content=[TextContent(type="text", text=text)])

    return server
