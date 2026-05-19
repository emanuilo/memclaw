"""First-run setup wizard and `memclaw configure` handler."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

ENV_FILE = Path.home() / ".memclaw" / ".env"

# Keys in the order they are prompted.
# `channel` is None for always-asked keys, or a channel name (e.g. "telegram")
# for keys that are only relevant to that bot command.
# `required` for a channel-scoped key means "required when invoked via that
# channel" (e.g. SLACK_BOT_TOKEN is required during `memclaw slack`, but not
# enforced during `memclaw configure` which shows everything).
# The Claude auth credential is chosen by the user at runtime; both candidates
# are listed here so reconfigure can render the right prompt.
_CLAUDE_SUBSCRIPTION_KEY = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "Claude subscription OAuth token (run `claude setup-token`)",
    True,
    None,
)
_CLAUDE_API_KEY = (
    "ANTHROPIC_API_KEY",
    "Anthropic API key (sk-ant-...)",
    True,
    None,
)

KEYS: list[tuple[str, str, bool, str | None]] = [
    ("OPENAI_API_KEY", "OpenAI API key (for embeddings + voice transcription)", True, None),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token", True, "telegram"),
    ("ALLOWED_USER_IDS", "Allowed Telegram user IDs (comma-separated)", True, "telegram"),
    ("SLACK_BOT_TOKEN", "Slack bot token (xoxb-...)", True, "slack"),
    ("SLACK_APP_TOKEN", "Slack app-level token for Socket Mode (xapp-...)", True, "slack"),
    ("SLACK_ALLOWED_CHANNELS", "Allowed Slack channel IDs (comma-separated)", False, "slack"),
    ("SLACK_ALLOWED_USERS", "Allowed Slack user IDs (comma-separated)", False, "slack"),
]


def _prompt_auth_mode(existing: dict[str, str]) -> str:
    """Ask the user how they want to authenticate with Claude.

    Returns "subscription" or "api_key".
    """
    console.print()
    console.print(
        Panel(
            "[bold]1)[/bold] Claude subscription (Pro / Max / Team)\n"
            "    No per-message cost — uses your subscription quota.\n"
            "    Generate a token with: [bold]claude setup-token[/bold]\n\n"
            "[bold]2)[/bold] Anthropic API key (pay-as-you-go)\n"
            "    Billed per token against your console credits.\n"
            "    Get a key at: console.anthropic.com",
            title="How do you want to authenticate with Claude?",
            border_style="bright_cyan",
        )
    )

    # Default to whichever the user already has configured.
    if existing.get("ANTHROPIC_API_KEY") and not existing.get("CLAUDE_CODE_OAUTH_TOKEN"):
        default = "2"
    else:
        default = "1"

    choice = Prompt.ask("Choose", choices=["1", "2"], default=default)
    return "subscription" if choice == "1" else "api_key"


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

    auth_mode = _prompt_auth_mode(existing)
    auth_key = (
        _CLAUDE_SUBSCRIPTION_KEY if auth_mode == "subscription" else _CLAUDE_API_KEY
    )
    # Drop the unselected credential so a stale value from a prior run can't
    # silently override the chosen mode at runtime.
    other_env_key = (
        _CLAUDE_API_KEY[0] if auth_mode == "subscription" else _CLAUDE_SUBSCRIPTION_KEY[0]
    )
    values.pop(other_env_key, None)

    # Prompt the chosen Claude credential first (right after the auth-mode
    # choice the user just made), then everything else.
    keys_to_prompt: list[tuple[str, str, bool, str | None]] = [auth_key, *KEYS]

    # A channel-scoped required key is only enforced when invoked via that
    # channel; in reconfigure mode nothing is enforced (user is just editing).
    def _is_required(required: bool, key_channel: str | None) -> bool:
        if reconfigure or not required:
            return False
        return key_channel is None or key_channel == channel

    for env_key, label, required, key_channel in keys_to_prompt:
        # Skip channel-scoped keys that don't match this invocation (unless
        # the user is explicitly reconfiguring, in which case show all).
        if not reconfigure and key_channel is not None and key_channel != channel:
            continue

        current = existing.get(env_key, "")
        masked = _mask(current)
        is_required = _is_required(required, key_channel)

        if reconfigure and current:
            prompt_text = f"{label} [{masked}]"
        elif is_required:
            prompt_text = f"{label} (required)"
        else:
            prompt_text = f"{label} (optional)"

        answer = Prompt.ask(prompt_text, default="", show_default=False)

        if answer:
            values[env_key] = answer
        elif current:
            values[env_key] = current

    # Validate required keys (always-required + channel-scoped required).
    for env_key, label, required, key_channel in keys_to_prompt:
        if _is_required(required, key_channel) and not values.get(env_key):
            console.print(f"[red]Error:[/red] {label} is required.")
            raise SystemExit(1)

    # Write to ~/.memclaw/.env
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")

    # Sync the current process env with the freshly-written .env. Without this,
    # MemclawConfig.__post_init__ would still see stale values (e.g. an OAuth
    # token left over from a prior run after the user switched to api_key).
    os.environ.pop(other_env_key, None)
    for k, v in values.items():
        if v:
            os.environ[k] = v

    console.print(f"\n[green]Config saved to {ENV_FILE}[/green]")
