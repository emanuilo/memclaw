"""First-run setup wizard and `memclaw configure` handler."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .backends import DEFAULT_BACKEND, get_backend_class, list_backends

console = Console()

ENV_FILE = Path.home() / ".memclaw" / ".env"

# Wordmark generated with pyfiglet font "ansi_shadow". Split into two halves
# so each can be colored independently (white for "mem", cyan for "claw")
# to mirror the logo's color split. Regenerate with:
#   python -c "import pyfiglet; print(pyfiglet.figlet_format('mem', font='ansi_shadow'))"
#   python -c "import pyfiglet; print(pyfiglet.figlet_format('claw', font='ansi_shadow'))"
_LOGO_MEM = (
    "███╗   ███╗███████╗███╗   ███╗\n"
    "████╗ ████║██╔════╝████╗ ████║\n"
    "██╔████╔██║█████╗  ██╔████╔██║\n"
    "██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║\n"
    "██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║\n"
    "╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝"
)
_LOGO_CLAW = (
    " ██████╗██╗      █████╗ ██╗    ██╗\n"
    "██╔════╝██║     ██╔══██╗██║    ██║\n"
    "██║     ██║     ███████║██║ █╗ ██║\n"
    "██║     ██║     ██╔══██║██║███╗██║\n"
    "╚██████╗███████╗██║  ██║╚███╔███╔╝\n"
    " ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ "
)


def _build_logo_banner() -> Text:
    """Assemble the two-color wordmark as a single Rich Text."""
    mem_lines = _LOGO_MEM.split("\n")
    claw_lines = _LOGO_CLAW.split("\n")
    banner = Text()
    for i, (m, c) in enumerate(zip(mem_lines, claw_lines)):
        banner.append(m, style="white")
        banner.append(c, style="cyan")
        if i < len(mem_lines) - 1:
            banner.append("\n")
    return banner

# Generic keys prompted for every install. The agent-backend credential is
# collected by the chosen backend's `wizard_setup()`, not by this list.
# `channel` is None for always-asked keys, or a channel name (e.g. "telegram")
# for keys that are only relevant to that bot command. `required` for a
# channel-scoped key means "required when invoked via that channel" (e.g.
# SLACK_BOT_TOKEN is required during `memclaw slack`, but not enforced during
# `memclaw configure` which shows everything).
KEYS: list[tuple[str, str, bool, str | None]] = [
    ("OPENAI_API_KEY", "OpenAI API key (for embeddings + voice transcription)", True, None),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token", True, "telegram"),
    ("ALLOWED_USER_IDS", "Allowed Telegram user IDs (comma-separated)", True, "telegram"),
    ("SLACK_BOT_TOKEN", "Slack bot token (xoxb-...)", True, "slack"),
    ("SLACK_APP_TOKEN", "Slack app-level token for Socket Mode (xapp-...)", True, "slack"),
    ("SLACK_ALLOWED_CHANNELS", "Allowed Slack channel IDs (comma-separated)", False, "slack"),
    ("SLACK_ALLOWED_USERS", "Allowed Slack user IDs (comma-separated)", False, "slack"),
]


def _select_backend(existing: dict[str, str]) -> str:
    """Pick the agent backend.

    With one registered backend this just returns its name silently.
    A panel only appears when there are multiple to choose from, so adding
    a second backend later becomes a UI change for free.
    """
    backends = list_backends()
    if len(backends) <= 1:
        return backends[0].name if backends else DEFAULT_BACKEND

    console.print()
    bullets = "\n\n".join(
        f"[bold]{i + 1})[/bold] {cls.display_name}" for i, cls in enumerate(backends)
    )
    console.print(
        Panel(bullets, title="Which agent SDK do you want to use?",
              border_style="bright_cyan")
    )

    # Default to whichever backend the existing config already names.
    current = existing.get("AGENT_BACKEND", "")
    default_idx = next(
        (str(i + 1) for i, cls in enumerate(backends) if cls.name == current),
        "1",
    )
    choices = [str(i + 1) for i in range(len(backends))]
    choice = Prompt.ask("Choose", choices=choices, default=default_idx)
    return backends[int(choice) - 1].name


def _mask(value: str) -> str:
    """Return a masked version of a secret for display."""
    if not value or len(value) < 8:
        return ""
    return value[:4] + "..." + value[-4:]


def _masked_input(prompt_text: str, *, visible: int = 4) -> str:
    """Read a line from stdin, echoing only the first `visible` chars verbatim
    and '*' for everything after. Used for API keys and other config values
    during the wizard so pasted secrets don't sit in the scrollback in clear.
    """
    import sys

    console.print(prompt_text, end=": ")
    sys.stdout.flush()

    if not sys.stdin.isatty():
        line = sys.stdin.readline().rstrip("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        return line

    try:
        import termios
        import tty
    except ImportError:
        # Non-POSIX (e.g. Windows): fall back to unmasked input.
        return input()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch in ("\x7f", "\b"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ord(ch) < 32:
                continue
            buf.append(ch)
            display = ch if len(buf) <= visible else "*"
            sys.stdout.write(display)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(buf)


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

    console.print()
    console.print(_build_logo_banner(), soft_wrap=True)
    console.print()

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
                "Let's set up your API tokens.\n"
                "Optional keys can be left blank and configured later\n"
                "with [bold]memclaw configure[/bold].",
                title="memclaw setup",
                border_style="bright_cyan",
            )
        )

    # Start from any previously-saved values so channel-scoped keys that we
    # skip this round are preserved.
    values: dict[str, str] = dict(existing)

    # 1) Choose backend, 2) let it collect its own credentials.
    backend_name = _select_backend(existing)
    values["AGENT_BACKEND"] = backend_name
    backend_cls = get_backend_class(backend_name)
    backend_values, drop_keys = backend_cls.wizard_setup(console, existing)
    values.update(backend_values)
    for key in drop_keys:
        values.pop(key, None)

    # A channel-scoped required key is only enforced when invoked via that
    # channel; in reconfigure mode nothing is enforced (user is just editing).
    def _is_required(required: bool, key_channel: str | None) -> bool:
        if reconfigure or not required:
            return False
        return key_channel is None or key_channel == channel

    for env_key, label, required, key_channel in KEYS:
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

        answer = _masked_input(prompt_text)

        if answer:
            values[env_key] = answer
        elif current:
            values[env_key] = current

    # Validate required keys (always-required + channel-scoped required).
    for env_key, label, required, key_channel in KEYS:
        if _is_required(required, key_channel) and not values.get(env_key):
            console.print(f"[red]Error:[/red] {label} is required.")
            raise SystemExit(1)

    # Write to ~/.memclaw/.env
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")

    # Sync the current process env with the freshly-written .env. Without this,
    # MemclawConfig.__post_init__ would still see stale values (e.g. an OAuth
    # token left over from a prior run after the user switched backends).
    for key in drop_keys:
        os.environ.pop(key, None)
    for k, v in values.items():
        if v:
            os.environ[k] = v

    console.print(f"\n[green]Config saved to {ENV_FILE}[/green]")
