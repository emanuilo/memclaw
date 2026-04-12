"""First-run setup wizard and `memclaw configure` handler."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

ENV_FILE = Path.home() / ".memclaw" / ".env"

# Keys in the order they are prompted
KEYS = [
    ("OPENAI_API_KEY", "OpenAI API key", True),
    ("ANTHROPIC_API_KEY", "Anthropic API key", True),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token", False),
    ("ALLOWED_USER_IDS", "Allowed Telegram user IDs (comma-separated)", False),
    ("WHATSAPP_ALLOWED_NUMBERS", "Extra WhatsApp numbers allowed to message the bot (comma-separated, e.g. +1234567890). Leave blank for self-notes only.", False),
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


def run_setup(*, reconfigure: bool = False) -> None:
    """Run the interactive setup wizard.

    Args:
        reconfigure: If True, show existing values and allow updating.
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

    values: dict[str, str] = {}

    for env_key, label, required in KEYS:
        current = existing.get(env_key, "")
        masked = _mask(current)

        if reconfigure and current:
            default_display = masked
            prompt_text = f"{label} [{default_display}]"
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
