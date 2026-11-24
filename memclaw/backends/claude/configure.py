"""The Claude backend's questions in `memclaw configure`: how to
authenticate, the credential, then the model and effort level. One function
per question, called in that order by `ask_claude_settings`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...terminal_input import choose, masked_input
from ..model_selection import ModelInfo
from .models import fetch_models
from .settings import DEFAULT_EFFORT, DEFAULT_MODEL

if TYPE_CHECKING:
    from rich.console import Console

    from ...config import ClaudeSettings, MemclawConfig


@dataclass(frozen=True)
class _Credential:
    """One way to authenticate: the env var it is saved in, how it is
    asked for, and the other credential's env var, dropped when it's saved."""

    auth_mode: str
    env_key: str
    label: str
    drop_key: str


_SUBSCRIPTION = _Credential(
    auth_mode="subscription",
    env_key="CLAUDE_CODE_OAUTH_TOKEN",
    label="Claude subscription OAuth token (run `claude setup-token`)",
    drop_key="ANTHROPIC_API_KEY",
)
_API_KEY = _Credential(
    auth_mode="api_key",
    env_key="ANTHROPIC_API_KEY",
    label="Anthropic API key (sk-ant-...)",
    drop_key="CLAUDE_CODE_OAUTH_TOKEN",
)


def ask_claude_settings(
    console: "Console", existing: dict[str, str], config: "MemclawConfig",
) -> tuple[dict[str, str], list[str]]:
    """Ask whether to use a subscription or API key, then prompt the
    chosen credential. Returns (values_to_save, env_keys_to_drop).
    """
    credential = _ask_auth_mode(console, existing)
    value = _ask_credential(console, existing, credential)

    values = {credential.env_key: value}
    drop_keys = [credential.drop_key]

    model_values, model_drops = ask_model_and_effort(
        console, existing, auth_mode=credential.auth_mode, credential=value,
        settings=config.claude,
    )
    values.update(model_values)
    drop_keys.extend(model_drops)
    return values, drop_keys


def _ask_auth_mode(console: "Console", existing: dict[str, str]) -> _Credential:
    if existing.get("ANTHROPIC_API_KEY") and not existing.get("CLAUDE_CODE_OAUTH_TOKEN"):
        default = 1
    else:
        default = 0
    choice = choose(
        console,
        title="How do you want to authenticate with Claude?",
        rows=[
            "Claude subscription (Pro / Max / Team)\n"
            "    No per-message cost — uses your subscription quota.\n"
            "    Generate a token with: [bold]claude setup-token[/bold]",
            "Anthropic API key (pay-as-you-go)\n"
            "    Billed per token against your console credits.\n"
            "    Get a key at: console.anthropic.com",
        ],
        default=default,
        separator="\n\n",
    )
    return _SUBSCRIPTION if choice == 0 else _API_KEY


def _ask_credential(console: "Console", existing: dict[str, str], credential: _Credential) -> str:
    """The credential typed, or the saved one on Enter; exits without one."""
    current = existing.get(credential.env_key, "")
    answer = masked_input(console, f"{credential.label} (required)")
    value = answer or current
    if not value:
        console.print(f"[red]Error:[/red] {credential.label} is required.")
        raise SystemExit(1)
    return value


def ask_model_and_effort(
    console: "Console",
    existing: dict[str, str],
    *,
    auth_mode: str,
    credential: str,
    settings: "ClaudeSettings",
) -> tuple[dict[str, str], list[str]]:
    """Ask which model to run and, if it supports one, which effort level.

    Returns (values_to_save, env_keys_to_drop). The list is fetched from
    Anthropic rather than hardcoded, so a newly released model shows up
    here with no code change.

    Every failure path returns empty, which leaves whatever the user
    already had in place: picking a model is a convenience, and it must
    never be the reason `memclaw configure` cannot finish.
    """
    # Like CURSOR_MODEL, a value exported only in the shell still counts
    # as the current one, so accepting the defaults never replaces it.
    current_model = (existing.get("CLAUDE_MODEL") or settings.model).strip() or DEFAULT_MODEL
    current_effort = (existing.get("CLAUDE_EFFORT") or settings.effort).strip().lower()

    def _keep_current(reason: str) -> tuple[dict[str, str], list[str]]:
        console.print(
            f"[yellow]Could not fetch the model list ({reason}) — "
            f"keeping [bold]{current_model}[/bold].[/yellow]"
        )
        return {}, []

    try:
        with console.status("[cyan]Fetching available Claude models...[/cyan]"):
            models = asyncio.run(fetch_models(auth_mode, credential))
    except Exception as exc:
        return _keep_current(f"{type(exc).__name__}: {exc}")

    if not models:
        return _keep_current("the API returned none")

    picked = _pick_model(console, models, current_model)
    if picked is None:
        # Kept a current model the list doesn't name. Its effort support
        # is unknown, so leave both settings exactly as they were.
        return {}, []
    values = {"CLAUDE_MODEL": picked.id}

    if not picked.effort_levels:
        # This model takes no effort value. Skip the question with no
        # message, and drop any level left over from a previous choice so
        # it can't be sent to a model that would reject it.
        return values, ["CLAUDE_EFFORT"]

    values["CLAUDE_EFFORT"] = _pick_effort(
        console, picked.effort_levels, current_effort,
    )
    return values, []

def _pick_model(
    console: "Console", models: list[ModelInfo], current_model: str,
) -> ModelInfo | None:
    """Show the numbered model picker, preselecting *current_model*.

    A current model the list doesn't name (an alias like `opus`, or an id
    this credential can't list) gets its own row at the end and stays the
    default, so pressing Enter never changes the model. Picking that row
    returns None.
    """
    current = "  [dim](current)[/dim]"
    rows = [
        f"{m.display_name}  [dim]{m.id}[/dim]{current if m.id == current_model else ''}"
        for m in models
    ]
    ids = [m.id for m in models]
    if current_model in ids:
        default = ids.index(current_model)
    else:
        rows.append(f"{current_model}{current}")
        default = len(models)

    index = choose(
        console, title="Which Claude model?", rows=rows, default=default,
    )
    return models[index] if index < len(models) else None

def _pick_effort(
    console: "Console", levels: list[str], current_effort: str,
) -> str:
    """Show the effort picker for a model that supports one.

    *levels* only ever holds levels this model accepts, so the preselected
    value falls back to the first one when the model has no `medium`.
    """
    preferred = current_effort if current_effort in levels else DEFAULT_EFFORT
    index = choose(
        console,
        title="Effort level?",
        rows=levels,
        default=levels.index(preferred) if preferred in levels else 0,
        footer=(
            "[dim]How deeply Claude thinks before answering. "
            "Lower is faster and cheaper.[/dim]"
        ),
        separator="    ",
    )
    return levels[index]
