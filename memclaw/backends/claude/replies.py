"""Talking to a connected Claude CLI: what a user message is sent as (text,
or the stream-json image protocol), and how the messages that come back
become a turn's result or a one-shot answer."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from loguru import logger

from ..base import ProgressCallback, TurnResult, report_tool_step
from ..mcp_tools import MCP_SERVER_NAME

# Claude sees Memclaw's tools as mcp__memclaw__<name>.
_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


def compose_prompt(context: str, user_message: str) -> str:
    """The per-turn context block goes at the top of the user message, so
    the session's system prompt stays static and cacheable."""
    return f"{context}\n\n{user_message}" if context else user_message


async def _image_prompt_stream(
    message: str, image_b64: str, image_media_type: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield a single streaming-input user message containing an image + text.

    The Claude CLI's stream-json protocol expects Anthropic-style content
    blocks here, so we pass an "image" block with a base64 source followed
    by the user's text. Targets claude-agent-sdk 0.2.x.
    """
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": message},
            ],
        },
        "parent_tool_use_id": None,
    }


class _TurnReply:
    """One turn's reply, read message by message: the last assistant text,
    the tool calls (logged and reported as steps), and the final result's
    session, turn count, cost and token usage."""

    def __init__(self, on_tool: ProgressCallback | None) -> None:
        self.result = TurnResult(text="")
        self.last_text = ""
        self._steps = 0
        self._on_tool = on_tool

    async def on_assistant(self, msg: AssistantMessage) -> None:
        turn_text = ""
        for block in msg.content:
            if isinstance(block, TextBlock):
                turn_text += block.text
            elif isinstance(block, ToolUseBlock):
                await self._on_tool_use(block)
        if turn_text:
            self.last_text = turn_text

    async def _on_tool_use(self, block: ToolUseBlock) -> None:
        args_str = json.dumps(block.input, ensure_ascii=False)
        if len(args_str) > 300:
            args_str = args_str[:300] + "..."
        tool_name = block.name.removeprefix(_TOOL_PREFIX)
        logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)
        self._steps += 1
        await report_tool_step(self._on_tool, self._steps, tool_name, block.input)

    def on_result(self, msg: ResultMessage) -> None:
        result = self.result
        result.session_id = msg.session_id or None
        result.num_turns = msg.num_turns
        result.cost_usd = msg.total_cost_usd
        if msg.usage:
            usage = msg.usage
            result.input_tokens = usage.get("input_tokens", 0) or 0
            result.output_tokens = usage.get("output_tokens", 0) or 0
            result.cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
            result.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
        if msg.result and not self.last_text:
            self.last_text = msg.result


async def read_turn(
    client: ClaudeSDKClient,
    prompt: str,
    *,
    image_b64: str | None,
    image_media_type: str,
    on_tool: ProgressCallback | None = None,
) -> TurnResult:
    """Send one user message on a connected client and collect the reply."""
    if image_b64:
        await client.query(_image_prompt_stream(prompt, image_b64, image_media_type))
    else:
        await client.query(prompt)

    reply = _TurnReply(on_tool)
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            await reply.on_assistant(msg)
        elif isinstance(msg, ResultMessage):
            reply.on_result(msg)

    reply.result.text = reply.last_text
    return reply.result


class OneShotReply:
    """A one-shot call's answer: all its text, and the error, if any, the
    CLI reported along the way."""

    def __init__(self) -> None:
        self.text = ""
        self.error: str | None = None

    def on_assistant(self, msg: AssistantMessage) -> None:
        if msg.error:
            self.error = msg.error
        for block in msg.content:
            if isinstance(block, TextBlock):
                self.text += block.text

    def on_result(self, msg: ResultMessage) -> None:
        if msg.is_error:
            self.error = self.error or msg.subtype
        if msg.result and not self.text:
            self.text = msg.result

    def answer(self) -> str:
        # A failed call (network, auth, rate limit) still comes back as text,
        # e.g. "API Error: Connection refused"; it must never be taken as the
        # answer, or consolidation writes it over MEMORY.md.
        if self.error:
            raise RuntimeError(f"Claude one-shot failed ({self.error}): {self.text[:200]}")
        return self.text


async def read_one_shot(client: ClaudeSDKClient, user_message: str) -> OneShotReply:
    """Send *user_message* on *client* and read the whole reply."""
    await client.query(user_message)
    reply = OneShotReply()
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            reply.on_assistant(msg)
        elif isinstance(msg, ResultMessage):
            reply.on_result(msg)
    return reply
