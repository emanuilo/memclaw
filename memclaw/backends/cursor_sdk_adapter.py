"""Adapters for Cursor SDK stream messages and tool-call naming."""

from __future__ import annotations

import json
from typing import Any, Mapping

from loguru import logger

from .base import TurnResult
from .mcp_tools import MCP_SERVER_NAME


def normalize_tool_call(name: str, args: Any) -> tuple[str, Any]:
    """Map Cursor SDK MCP wrapper calls to Memclaw tool names for logging."""
    if isinstance(args, dict):
        if name.lower() == "mcp":
            inner_name = args.get("toolName") or args.get("tool_name")
            if inner_name:
                inner_args = args.get("args", args)
                return str(inner_name), inner_args
    prefix = f"{MCP_SERVER_NAME}_"
    if name.startswith(prefix):
        return name[len(prefix) :], args
    return name, args


def extract_run_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("result", "text", "output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def assistant_message_text(message: Any) -> str:
    """Concatenate all text blocks from a Cursor SDK assistant stream message."""
    content = getattr(getattr(message, "message", None), "content", ())
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
    return "".join(parts)


def _log_tool_call(name: str, args: Any) -> None:
    tool_name, tool_args = normalize_tool_call(name, args)
    if tool_args is None:
        args_str = "{}"
    else:
        try:
            args_str = json.dumps(tool_args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tool_args)
    if len(args_str) > 300:
        args_str = args_str[:300] + "..."
    logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)


def _log_tool_result(name: str, args: Any, *, status: str, result: Any) -> None:
    tool_name, _ = normalize_tool_call(name, args)
    if result is None:
        result_str = ""
    elif isinstance(result, str):
        result_str = result
    else:
        try:
            result_str = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            result_str = str(result)
    if len(result_str) > 300:
        result_str = result_str[:300] + "..."
    if status == "error":
        logger.warning(
            "Tool {name} failed: {result}",
            name=tool_name,
            result=result_str or "(no details)",
        )
    else:
        logger.info(
            "Tool {name} completed: {result}",
            name=tool_name,
            result=result_str or "(ok)",
        )


def _mapping_get_int(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _mapping_get_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def parse_cursor_usage(usage: Mapping[str, Any] | None) -> dict[str, int | float | None]:
    """Normalize Cursor SDK usage payloads to Memclaw TurnResult fields."""
    if not usage:
        return {}
    return {
        "input_tokens": _mapping_get_int(
            usage,
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
        ),
        "output_tokens": _mapping_get_int(
            usage,
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ),
        "cache_read_tokens": _mapping_get_int(
            usage,
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_read_tokens",
            "cacheReadTokens",
        ),
        "cache_creation_tokens": _mapping_get_int(
            usage,
            "cache_write_tokens",
            "cacheWriteTokens",
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_creation_tokens",
            "cacheCreationTokens",
        ),
        "cost_usd": _mapping_get_float(
            usage,
            "total_cost_usd",
            "totalCostUsd",
            "cost_usd",
            "costUsd",
        ),
    }


def accumulate_usage(
    totals: dict[str, int | float | None],
    usage: Mapping[str, Any] | None,
) -> None:
    """Merge one SDK usage payload into running token/cost totals."""
    parsed = parse_cursor_usage(usage)
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"):
        value = parsed.get(key)
        if isinstance(value, int) and value:
            totals[key] = int(totals.get(key) or 0) + value
    cost = parsed.get("cost_usd")
    if isinstance(cost, float):
        totals["cost_usd"] = float(totals.get("cost_usd") or 0.0) + cost


def _empty_usage_totals() -> dict[str, int | float | None]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "token_delta_sum": 0,
        "cost_usd": None,
    }


def record_interaction_usage(
    totals: dict[str, int | float | None],
    update: Any,
) -> None:
    """Record token usage from a Cursor SDK InteractionUpdate."""
    update_type = getattr(update, "type", None)
    if update_type == "turn-ended":
        totals["turn_count"] = int(totals.get("turn_count") or 0) + 1
        usage = getattr(update, "usage", None)
        if isinstance(usage, Mapping):
            accumulate_usage(totals, usage)
        return
    if update_type == "token-delta":
        tokens = getattr(update, "tokens", 0) or 0
        try:
            delta = int(tokens)
        except (TypeError, ValueError):
            return
        if delta:
            totals["token_delta_sum"] = int(totals.get("token_delta_sum") or 0) + delta


class RunUsageTracker:
    """Collect usage from Cursor SDK interaction deltas.

    The bridge only forwards ``turn-ended`` usage when ``SendOptions.on_delta``
    is set (which sets ``enableDeltas`` on the wire). Memclaw always attaches
    this tracker to agent sends so token accounting works.
    """

    def __init__(self) -> None:
        self.totals = _empty_usage_totals()
        self.totals["turn_count"] = 0

    def on_delta(self, update: Any) -> None:
        record_interaction_usage(self.totals, update)

    def apply_to_result(self, result: TurnResult) -> None:
        input_tokens = int(self.totals["input_tokens"] or 0)
        output_tokens = int(self.totals["output_tokens"] or 0)
        token_delta_sum = int(self.totals["token_delta_sum"] or 0)
        if not output_tokens and token_delta_sum:
            output_tokens = token_delta_sum

        result.input_tokens = input_tokens
        result.output_tokens = output_tokens
        result.cache_read_tokens = int(self.totals["cache_read_tokens"] or 0)
        result.cache_creation_tokens = int(self.totals["cache_creation_tokens"] or 0)
        cost = self.totals["cost_usd"]
        if isinstance(cost, float):
            result.cost_usd = cost
        turn_count = int(self.totals.get("turn_count") or 0)
        if turn_count > result.num_turns:
            result.num_turns = turn_count


def _handle_sdk_message(message: Any, *, pending_tools: dict[str, str]) -> str:
    """Process one SDK message for logging; return latest assistant text."""
    msg_type = getattr(message, "type", None)
    if msg_type == "assistant":
        turn_text = assistant_message_text(message)
        if turn_text:
            return turn_text
        return ""
    if msg_type != "tool_call":
        return ""

    status = str(getattr(message, "status", ""))
    name = getattr(message, "name", "")
    args = getattr(message, "args", None)
    call_id = str(getattr(message, "call_id", "") or name)
    if status == "running":
        _log_tool_call(name, args)
        pending_tools[call_id] = name
    elif status in {"completed", "error"}:
        pending_tools.pop(call_id, None)
        _log_tool_result(
            name,
            args,
            status=status,
            result=getattr(message, "result", None),
        )
    elif status:
        logger.warning(
            "Tool {name} unexpected status {status!r}",
            name=normalize_tool_call(name, args)[0],
            status=status,
        )
    return ""


async def collect_run_result(
    run: Any,
    *,
    max_turns: int,
    usage_tracker: RunUsageTracker | None = None,
) -> TurnResult:
    """Drain the run stream for logging and return a normalized TurnResult."""
    last_text = ""
    pending_tools: dict[str, str] = {}

    async for event in run.events():
        message = getattr(event, "sdk_message", None)
        if message is not None:
            turn_text = _handle_sdk_message(message, pending_tools=pending_tools)
            if turn_text:
                last_text = turn_text

    for call_id, name in pending_tools.items():
        tool_name, _ = normalize_tool_call(name, {})
        logger.warning(
            "Tool {name} started but never completed (call_id={call_id}). "
            "It may have been blocked by a Cursor hook or failed before MCP execution.",
            name=tool_name,
            call_id=call_id or "(unknown)",
        )

    wait_result = await run.wait()
    wait_text = extract_run_text(wait_result)
    if len(wait_text) > len(last_text):
        last_text = wait_text
    elif not last_text:
        last_text = wait_text

    reported_turns = getattr(wait_result, "num_turns", 0) or 0
    num_turns = max(
        int(getattr(usage_tracker, "totals", {}).get("turn_count") or 0) if usage_tracker else 0,
        reported_turns,
        1,
    )
    if num_turns > max_turns:
        logger.warning(
            "Cursor SDK reported {reported} turns (Memclaw max_turns={max}); "
            "the SDK does not expose a turn cap — tighten prompts or hooks if needed.",
            reported=num_turns,
            max=max_turns,
        )

    result = TurnResult(text=last_text, num_turns=num_turns)
    if usage_tracker is not None:
        usage_tracker.apply_to_result(result)
    return result
