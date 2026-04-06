"""First-run setup wizard and `memclaw configure` handler."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

ENV_FILE = Path.home() / ".memclaw" / ".env"

# Keys in the order they are prompted.
# `channel` is None for always-asked keys, or a channel name (e.g. "telegram")
# for keys that are only relevant to that bot command.
KEYS: list[tuple[str, str, bool, str | None]] = [
    ("OPENAI_API_KEY", "OpenAI API key", True, None),
    ("ANTHROPIC_API_KEY", "Anthropic API key", True, None),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token", False, "telegram"),
    ("ALLOWED_USER_IDS", "Allowed Telegram user IDs (comma-separated)", False, "telegram"),
    ("SLACK_BOT_TOKEN", "Slack bot token (xoxb-...)", False, "slack"),
    ("SLACK_APP_TOKEN", "Slack app-level token for Socket Mode (xapp-...)", False, "slack"),
    ("SLACK_ALLOWED_CHANNELS", "Allowed Slack channel IDs (comma-separated, optional)", False, "slack"),
]


def _mask(value: str) -> str:
    """Return a masked version of a secret for display."""
    if not value or len(value) < 8:
        return ""
    return value[:4] + "..." + value[-4:]


def _load_existing() -> dict[str, str]:
    """Load existing values from ~/.memclaw/.env."""
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def needs_setup() -> bool:
    """Return True if first-run setup is needed."""
    return not ENV_FILE.exists()


def run_setup(*, reconfigure: bool = False, channel: str | None = None) -> None:
    """Run the interactive setup wizard.

    Args:
        reconfigure: If True, show existing values and allow updating all keys.
        channel: If set (e.g. "telegram"), only prompt for always-asked keys
                 plus keys scoped to that channel. Ignored when reconfiguring.
    """
    existing = _load_existing()

    if reconfigure:
        console.print(
            Panel(
                "Update your Memclaw configuration.\n"
                "Press [bold]Enter[/bold] to keep the current value.",
                title="memclaw configure",
                border_style="bright_cyan",
            )
        )
    else:
        console.print(
            Panel(
                "[bold]Welcome to Memclaw![/bold]\n\n"
                "Let's set up your API keys.\n"
                "Optional keys can be left blank and configured later\n"
                "with [bold]memclaw configure[/bold].",
                title="memclaw setup",
                border_style="bright_cyan",
            )
        )

    # Start from any previously-saved values so channel-scoped keys that we
    # skip this round are preserved.
    values: dict[str, str] = dict(existing)

    for env_key, label, required, key_channel in KEYS:
        # Skip channel-scoped keys that don't match this invocation (unless
        # the user is explicitly reconfiguring, in which case show all).
        if not reconfigure and key_channel is not None and key_channel != channel:
            continue

        current = existing.get(env_key, "")
        masked = _mask(current)

        if reconfigure and current:
            prompt_text = f"{label} [{masked}]"
        elif required:
            prompt_text = f"{label} (required)"
        else:
            prompt_text = f"{label} (optional)"

        answer = Prompt.ask(prompt_text, default="", show_default=False)

        if answer:
            values[env_key] = answer
        elif current:
            values[env_key] = current

    # Validate required keys
    if not values.get("OPENAI_API_KEY"):
        console.print("[red]Error:[/red] OpenAI API key is required.")
        raise SystemExit(1)
    if not values.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error:[/red] Anthropic API key is required.")
        raise SystemExit(1)

    # Write to ~/.memclaw/.env
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")

    console.print(f"\n[green]Config saved to {ENV_FILE}[/green]")
