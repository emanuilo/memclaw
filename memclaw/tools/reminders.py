"""Tools for reminders: schedule one, list them, cancel one. They only work
in a messaging bot, which runs the scheduler and has a chat to send to."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from ..reminders import ReminderScheduler
from .context import Tool, ToolContext

REMINDER_CREATE: dict[str, Any] = {
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
}

REMINDER_LIST: dict[str, Any] = {
    "name": "reminder_list",
    "description": "List the user's pending reminders.",
    "input_schema": {"type": "object", "properties": {}},
}

REMINDER_CANCEL: dict[str, Any] = {
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
}


def _reminder_chat(ctx: ToolContext) -> tuple[ReminderScheduler, str, str] | str:
    """(scheduler, platform, chat_id) for this chat's reminders, or why
    there are none here."""
    if ctx.scheduler is None:
        return "Reminders are only available when running in a messaging bot (telegram/whatsapp/slack)."
    if not ctx.platform or not ctx.chat_id:
        return "Reminders require a messaging context (missing platform/chat_id)."
    return ctx.scheduler, ctx.platform, ctx.chat_id


def _first_fire(fire_at_str: str | None, interval: int | None) -> datetime | str:
    """When the reminder first fires (local time), or why it can't be set."""
    if fire_at_str:
        try:
            fire_at = datetime.fromisoformat(fire_at_str)
        except ValueError:
            return f"Invalid fire_at: {fire_at_str!r}. Use local ISO 8601."
        if fire_at.tzinfo is not None:
            fire_at = fire_at.astimezone().replace(tzinfo=None)
        return fire_at
    if interval:
        return datetime.now() + timedelta(seconds=interval)
    return "Provide either fire_at or interval_seconds (or both)."


async def reminder_create(ctx: ToolContext, args: dict) -> str:
    chat = _reminder_chat(ctx)
    if isinstance(chat, str):
        return chat
    scheduler, platform, chat_id = chat

    text: str = args["text"].strip()
    interval: int | None = args.get("interval_seconds")
    fire_at = _first_fire(args.get("fire_at"), interval)
    if isinstance(fire_at, str):
        return fire_at

    if interval is not None and interval < 60:
        return "interval_seconds must be at least 60."

    rid = scheduler.create(
        platform=platform,
        chat_id=chat_id,
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


async def reminder_list(ctx: ToolContext, args: dict) -> str:
    chat = _reminder_chat(ctx)
    if isinstance(chat, str):
        return chat
    scheduler, platform, chat_id = chat
    items = scheduler.list_for(platform, chat_id)
    if not items:
        return "No pending reminders."
    lines = []
    for it in items:
        line = f"#{it['id']} at {it['fire_at']}: {it['text']}"
        if it["interval_seconds"]:
            line += f" (every {it['interval_seconds']}s)"
        lines.append(line)
    return "\n".join(lines)


async def reminder_cancel(ctx: ToolContext, args: dict) -> str:
    chat = _reminder_chat(ctx)
    if isinstance(chat, str):
        return chat
    scheduler, platform, chat_id = chat
    rid = int(args["id"])
    ok = scheduler.cancel(rid, platform=platform, chat_id=chat_id)
    return f"Reminder #{rid} cancelled." if ok else f"No pending reminder #{rid}."


TOOLS = [
    Tool(REMINDER_CREATE, reminder_create, "Setting a reminder", "text"),
    Tool(REMINDER_LIST, reminder_list, "Checking reminders"),
    Tool(REMINDER_CANCEL, reminder_cancel, "Cancelling a reminder"),
]
