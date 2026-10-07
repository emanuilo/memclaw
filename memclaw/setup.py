"""First-run setup wizard and `memclaw configure` handler."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .backends import DEFAULT_BACKEND, get_backend_class, list_backends
from .prompts import choose

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


def build_logo_banner() -> Text:
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


def print_logo() -> None:
    """Print the two-color wordmark to the shared console."""
    console.print()
    console.print(build_logo_banner(), soft_wrap=True)
    console.print()


# Supported front-end platforms in display order. The first one is also the
# default when nothing has been configured yet.
PLATFORMS: list[tuple[str, str]] = [
    ("telegram", "Telegram"),
    ("whatsapp", "WhatsApp"),
    ("slack", "Slack"),
    ("terminal", "Terminal chat"),
]

# Generic keys prompted for every install. The agent-backend credential is
# collected by the chosen backend's `wizard_setup()`, not by this list.
# `platform` is None for always-asked keys, or a platform name for keys that
# are only relevant when that platform is selected. `required` for a
# platform-scoped key is enforced only when that platform is the active one.
KEYS: list[tuple[str, str, bool, str | None]] = [
    ("OPENAI_API_KEY", "OpenAI API key (for embeddings + voice transcription)", True, None),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token", True, "telegram"),
    ("ALLOWED_USER_IDS", "Allowed Telegram user IDs (comma-separated)", True, "telegram"),
    ("SLACK_BOT_TOKEN", "Slack bot token (xoxb-...)", True, "slack"),
    ("SLACK_APP_TOKEN", "Slack app-level token for Socket Mode (xapp-...)", True, "slack"),
    ("SLACK_ALLOWED_CHANNELS", "Allowed Slack channel IDs (comma-separated)", False, "slack"),
    ("SLACK_ALLOWED_USERS", "Allowed Slack user IDs (comma-separated)", False, "slack"),
]


def _select_platform(existing: dict[str, str]) -> str:
    """Pick the front-end platform Memclaw should launch."""
    current = existing.get("MEMCLAW_PLATFORM", "")
    default = next(
        (i for i, (name, _) in enumerate(PLATFORMS) if name == current), 0,
    )
    index = choose(
        console,
        title="How do you want to talk to Memclaw?",
        rows=[label for _, label in PLATFORMS],
        default=default,
        footer=(
            "[dim]See the README for instructions on how to set up Telegram, "
            "WhatsApp, or Slack credentials.[/dim]"
        ),
        separator="\n\n",
    )
    return PLATFORMS[index][0]


def _select_backend(existing: dict[str, str]) -> str:
    """Pick the agent backend.

    With one registered backend this just returns its name silently.
    A panel only appears when there are multiple to choose from, so adding
    a second backend later becomes a UI change for free.
    """
    backends = list_backends()
    if len(backends) <= 1:
        return backends[0].name if backends else DEFAULT_BACKEND

    # Default to whichever backend the existing config already names.
    current = existing.get("AGENT_BACKEND", "")
    default = next(
        (i for i, cls in enumerate(backends) if cls.name == current), 0,
    )
    index = choose(
        console,
        title="Which agent SDK do you want to use?",
        rows=[cls.display_name for cls in backends],
        default=default,
        separator="\n\n",
    )
    return backends[index].name


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


def update_env_file(values: dict[str, str], drop_keys: list[str] | tuple[str, ...] = ()) -> None:
    """Merge *values* into ~/.memclaw/.env and remove *drop_keys*.

    Keys are written in their existing order, new ones appended; a key with
    an empty value is removed, matching how the wizard writes the file.
    """
    merged = _load_existing()
    merged.update(values)
    for key in drop_keys:
        merged.pop(key, None)
    _write_env(merged)


def _write_env(values: dict[str, str]) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in values.items() if v]
    ENV_FILE.write_text("\n".join(lines) + "\n")


def needs_setup() -> bool:
    """Return True if first-run setup is needed."""
    return not ENV_FILE.exists()


def run_setup(*, reconfigure: bool = False, memory_dir: Path | str | None = None) -> None:
    """Run the interactive setup wizard.

    Args:
        reconfigure: If True, frame the prompts as "update your config"
                     instead of "welcome". The set of keys asked is identical
                     either way -- it depends only on the chosen platform.
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
                "Let's set up your API tokens.\n"
                "Optional keys can be left blank and configured later\n"
                "with [bold]memclaw configure[/bold].",
                title="memclaw setup",
                border_style="bright_cyan",
            )
        )

    # Start from any previously-saved values so platform-scoped keys we don't
    # prompt for this round (because the user picked a different platform) are
    # preserved and remain available if they switch back later.
    values: dict[str, str] = dict(existing)

    # 1) Choose backend, 2) let it collect its own credentials.
    backend_name = _select_backend(existing)
    values["AGENT_BACKEND"] = backend_name
    backend_cls = get_backend_class(backend_name)
    backend_values, drop_keys = backend_cls.wizard_setup(console, existing, memory_dir=memory_dir)
    values.update(backend_values)
    for key in drop_keys:
        values.pop(key, None)

    def _prompt_key(env_key: str, label: str, required: bool) -> None:
        current = existing.get(env_key, "")
        if current:
            prompt_text = f"{label} [{_mask(current)}]"
        elif required:
            prompt_text = f"{label} (required)"
        else:
            prompt_text = f"{label} (optional)"
        answer = _masked_input(prompt_text)
        if answer:
            values[env_key] = answer
        elif current:
            values[env_key] = current

    # 3) Ask for the always-required keys (currently just OpenAI) before the
    # platform picker, so the order is: Claude creds → OpenAI key → platform.
    for env_key, label, required, key_platform in KEYS:
        if key_platform is None:
            _prompt_key(env_key, label, required)

    # 4) Choose the front-end platform; this scopes which platform-specific
    # keys we'll prompt for below.
    platform = _select_platform(existing)
    values["MEMCLAW_PLATFORM"] = platform

    # 5) Ask for keys scoped to the chosen platform.
    for env_key, label, required, key_platform in KEYS:
        if key_platform == platform:
            _prompt_key(env_key, label, required)

    # Validate required keys (always-required + platform-scoped required).
    def _is_required(required: bool, key_platform: str | None) -> bool:
        if not required:
            return False
        return key_platform is None or key_platform == platform

    for env_key, label, required, key_platform in KEYS:
        if _is_required(required, key_platform) and not values.get(env_key):
            console.print(f"[red]Error:[/red] {label} is required.")
            raise SystemExit(1)

    # Write to ~/.memclaw/.env
    _write_env(values)

    # Sync the current process env with the freshly-written .env. Without this,
    # MemclawConfig.__post_init__ would still see stale values (e.g. an OAuth
    # token left over from a prior run after the user switched backends).
    for key in drop_keys:
        os.environ.pop(key, None)
    for k, v in values.items():
        if v:
            os.environ[k] = v

    console.print(f"\n[green]Config saved to {ENV_FILE}[/green]")

    # Verify OpenAI model access so the user finds out about disabled models
    # right now, not when their first voice message silently disappears.
    api_key = values.get("OPENAI_API_KEY")
    if api_key:
        import asyncio

        from .openai_health import print_probe_report, probe_openai

        console.print("\n[cyan]Checking OpenAI model access...[/cyan]")
        try:
            report = asyncio.run(probe_openai(api_key))
            print_probe_report(report, console)
        except Exception as exc:
            console.print(f"[yellow]Skipping model check:[/yellow] {exc}")
