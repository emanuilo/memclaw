"""The places Memclaw chats in: the chat apps and the terminal.

Each platform is one module here. The chat apps' (`telegram.py`,
`slack.py`, `whatsapp.py`) only do that app's I/O: receive a message,
download its media, send the reply. What happens in between (voice to text,
link summaries, the prompt, the agent call, reminders) is shared, in
`inbound.py`. `terminal.py` is the chat in the shell that runs `memclaw`.

Each platform module has a `run(config)` that starts it. The bare `memclaw`
command and the setup wizard find the platforms in `PLATFORMS` below, so
adding one is a new module plus one line there. A platform module (and its
chat SDK) is only imported when that platform is started.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    from ..config import MemclawConfig

console = Console()


@dataclass(frozen=True)
class Platform:
    """A place to chat in; its code is the module `memclaw/bot/<name>.py`."""

    name: str
    label: str  # shown by the setup wizard

    def run(self, config: MemclawConfig) -> None:
        """Start the bot and block until it stops."""
        import_module(f"{__name__}.{self.name}").run(config)


# In the order the setup wizard lists them.
PLATFORMS: dict[str, Platform] = {
    p.name: p
    for p in (
        Platform("telegram", "Telegram"),
        Platform("whatsapp", "WhatsApp"),
        Platform("slack", "Slack"),
        Platform("terminal", "Terminal chat"),
    )
}

# What the bare `memclaw` starts when MEMCLAW_PLATFORM is empty or unknown.
DEFAULT_PLATFORM = "terminal"


def chosen_platform(config: MemclawConfig) -> Platform:
    """The platform *config* names (MEMCLAW_PLATFORM), else the terminal."""
    return PLATFORMS.get(config.platform) or PLATFORMS[DEFAULT_PLATFORM]


def status_rows(config: MemclawConfig) -> list[tuple[str, object]]:
    """``(label, value)`` pairs `memclaw status` shows for the platform."""
    return [("Platform", config.platform or DEFAULT_PLATFORM)]


def mask_user_id(user_id: str | int) -> str:
    """Mask the middle of a user identifier for log safety.

    One `*` per hidden character so length is preserved.
    Example: '100951236' -> '100****36'.
    """
    s = str(user_id)
    if len(s) <= 5:
        return "*" * len(s)
    return f"{s[:3]}{'*' * (len(s) - 5)}{s[-2:]}"


def require_setting(value: str, env_var: str) -> None:
    """Exit with a hint unless *value* (the setting *env_var*) is set."""
    if not value:
        console.print(f"[red]Error:[/red] {env_var} is not set.")
        console.print("Run [bold]memclaw configure[/bold] to set it.")
        raise SystemExit(1)


def allowlist_label(allowlist: list, env_var: str) -> str:
    """How the start banner shows an allowlist. An empty one lets nobody in,
    so it also prints a warning naming the setting to fill in."""
    if allowlist:
        return ", ".join(str(who) for who in allowlist)
    console.print(
        f"[yellow]Warning:[/yellow] {env_var} is empty, so nobody can use the bot. "
        f"Set it to a comma-separated list of IDs with [bold]memclaw configure[/bold] "
        f"(or in ~/.memclaw/.env) and restart."
    )
    return "nobody"


def log_to(log_file: Path | None = None) -> None:
    """Log INFO to the terminal and, with a *log_file*, DEBUG to it (rotated)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )
    if log_file is not None:
        logger.add(str(log_file), rotation="10 MB", retention="7 days", level="DEBUG")
