"""The setup wizard: first run and `memclaw configure`."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..backends import DEFAULT_BACKEND, get_backend_class, list_backends
from ..config import (
    ENV_FILE,
    MemclawConfig,
    Setting,
    all_settings,
    read_env_file,
    set_environment,
    write_env_file,
)
from ..bot import PLATFORMS as CHAT_PLATFORMS
from ..terminal_input import choose, masked_input
from .doctor import check_openai

console = Console()

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


# Supported front-end platforms (name, label) in display order: the chat
# apps of memclaw.bot, then the terminal. The first one is also the default
# when nothing has been configured yet.
PLATFORMS: list[tuple[str, str]] = [
    *((p.name, p.label) for p in CHAT_PLATFORMS.values()),
    ("terminal", "Terminal chat"),
]


def questions(platform: str | None = None) -> list[Setting]:
    """The settings the wizard asks for (those declared with a `prompt=` in
    memclaw.config), in declaration order.

    With no *platform*: the ones asked on every install (their group isn't a
    platform, e.g. the OpenAI key). With a *platform*: the ones of that
    platform's group, asked only when it is chosen. The agent-backend
    credentials are asked by the chosen backend's `wizard_setup()` instead.
    """
    platforms = {name for name, _ in PLATFORMS}
    return [
        s for s in all_settings()
        if s.prompt and (s.group == platform if platform else s.group not in platforms)
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
            "[dim]Setup for Telegram, WhatsApp and Slack credentials: "
            "https://github.com/emanuilo/memclaw/blob/main/docs/platforms.md[/dim]"
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


def needs_setup() -> bool:
    """Return True if first-run setup is needed."""
    return not ENV_FILE.exists()


def run_setup(config: MemclawConfig, *, reconfigure: bool = False) -> None:
    """Run the interactive setup wizard and save the answers to ~/.memclaw/.env.

    Args:
        config: The active config (its memory dir, and the current values the
                backend's questions start from).
        reconfigure: If True, frame the prompts as "update your config"
                     instead of "welcome". The set of keys asked is identical
                     either way -- it depends only on the chosen platform.
    """
    existing = read_env_file()
    _print_intro(reconfigure)

    # Start from any previously-saved values so platform-scoped keys we don't
    # prompt for this round (because the user picked a different platform) are
    # preserved and remain available if they switch back later.
    values: dict[str, str] = dict(existing)

    # The order is: backend + its credentials -> OpenAI key -> platform -> its keys.
    drop_keys = _setup_backend(config, existing, values)
    _ask(questions(), existing, values)
    platform = _select_platform(existing)
    values["MEMCLAW_PLATFORM"] = platform
    _ask(questions(platform), existing, values)
    _require_answers(questions() + questions(platform), values)

    write_env_file(values)
    # Sync the current process env with the freshly-written .env. Without this,
    # a MemclawConfig made next would still see stale values (e.g. an OAuth
    # token left over from a prior run after the user switched backends).
    set_environment(values, drop_keys)
    console.print(f"\n[green]Config saved to {ENV_FILE}[/green]")

    check_openai(values.get("OPENAI_API_KEY"), console)


def _print_intro(reconfigure: bool) -> None:
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


def _setup_backend(config: MemclawConfig, existing: dict[str, str], values: dict[str, str]) -> list[str]:
    """Choose the backend and let it collect its own credentials into
    *values*. Returns the keys it wants removed."""
    backend_name = _select_backend(existing)
    values["AGENT_BACKEND"] = backend_name
    backend_cls = get_backend_class(backend_name)
    backend_values, drop_keys = backend_cls.wizard_setup(console, existing, config=config)
    values.update(backend_values)
    for key in drop_keys:
        values.pop(key, None)
    return drop_keys


def _ask(settings: list[Setting], existing: dict[str, str], values: dict[str, str]) -> None:
    """Ask for each of *settings*; Enter keeps the saved value."""
    for s in settings:
        assert s.env is not None and s.prompt is not None
        current = existing.get(s.env, "")
        if current:
            prompt_text = f"{s.prompt} [{_mask(current)}]"
        else:
            prompt_text = f"{s.prompt} ({'required' if s.required else 'optional'})"
        answer = masked_input(console, prompt_text)
        if answer:
            values[s.env] = answer
        elif current:
            values[s.env] = current


def _require_answers(settings: list[Setting], values: dict[str, str]) -> None:
    for s in settings:
        if s.required and s.env and not values.get(s.env):
            console.print(f"[red]Error:[/red] {s.prompt} is required.")
            raise SystemExit(1)
