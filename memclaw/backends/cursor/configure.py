"""Setting the Cursor backend up: its questions in `memclaw configure`, and
the tool-restriction hooks it installs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.prompt import Prompt

from ...terminal_input import masked_input
from .hooks import cursor_hooks_status, ensure_cursor_hooks

if TYPE_CHECKING:
    from rich.console import Console

    from ...config import MemclawConfig

# Scrub Claude credentials when switching to Cursor so they can't shadow selection.
_DROP_KEYS = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]


def ask_cursor_settings(
    console: "Console", existing: dict[str, str], config: "MemclawConfig",
) -> tuple[dict[str, str], list[str]]:
    """Ask for the API key and the model, and install the tool-restriction
    hooks. Returns (values_to_save, env_keys_to_drop)."""
    current = existing.get("CURSOR_API_KEY", config.cursor.api_key)
    answer = masked_input(console, "Cursor API key (required)")
    value = answer or current
    if not value:
        console.print("[red]Error:[/red] Cursor API key is required.")
        raise SystemExit(1)

    values: dict[str, str] = {"CURSOR_API_KEY": value}
    drop_keys = list(_DROP_KEYS)

    model_current = existing.get("CURSOR_MODEL", config.cursor.model)
    model_answer = Prompt.ask(
        r"Cursor model \[composer-2.5 is default, optional]",
        default="",
        show_default=False,
    )
    if model_answer.strip():
        values["CURSOR_MODEL"] = model_answer.strip()
        if model_answer.strip() != model_current:
            # An effort level picked with /effort belongs to the old model.
            drop_keys.append("CURSOR_EFFORT")
    elif model_current:
        values["CURSOR_MODEL"] = model_current

    install_hooks(config.memory_dir)
    return values, drop_keys


def install_hooks(memory_dir: Path, *, log_ready: bool = False) -> None:
    """Install or refresh the hooks that restrict the agent to Memclaw's
    tools, warning when they aren't ready (and, with *log_ready*, logging
    their status when they are)."""
    if ensure_cursor_hooks(memory_dir):
        if log_ready:
            status = cursor_hooks_status(memory_dir)
            logger.info("Cursor tool-restriction hooks: {status}", status=status)
        return
    status = cursor_hooks_status(memory_dir)
    logger.warning(
        "Cursor tool-restriction hooks are not ready ({status}). "
        "Built-in Cursor tools may be used until hooks are installed.",
        status=status,
    )
