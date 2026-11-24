"""The Claude backend's settings: which credential is configured, the
environment the CLI runs with, and the model and effort to use."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import process_environment
from .models import SDK_EFFORT_LEVELS

if TYPE_CHECKING:
    from ...config import MemclawConfig


DEFAULT_MODEL = "claude-sonnet-5-5"

# Memory consolidation: rare and one-shot, so a stronger model is worth it.
DEFAULT_CONSOLIDATION_MODEL = "claude-opus-5-5"
DEFAULT_CONSOLIDATION_EFFORT = "medium"

# Used when CLAUDE_EFFORT is unset and the default model runs, and
# preselected in the wizard for any model that supports effort at all.
DEFAULT_EFFORT = "medium"


def claude_auth_mode(config: "MemclawConfig") -> str:
    """Which Claude credential is configured.

    Returns "subscription" when CLAUDE_CODE_OAUTH_TOKEN is set (billed
    against the Claude plan, no per-message cost), "api_key" when only
    ANTHROPIC_API_KEY is set, or "" when neither is set.
    """
    if config.claude.oauth_token:
        return "subscription"
    if config.claude.api_key:
        return "api_key"
    return ""


def resolve_model(config: "MemclawConfig") -> str:
    """The configured model, or the built-in default when unset."""
    return (config.claude.model or "").strip() or DEFAULT_MODEL


def resolve_effort(config: "MemclawConfig") -> str | None:
    """The configured effort level, or None to let the model decide.

    Anything the SDK does not recognise is treated as unset: ~/.memclaw/.env
    is a plain text file, and a typo there should not reach the CLI as an
    argument it will reject. An unset level means `DEFAULT_EFFORT` on the
    default model and None (the SDK omits the flag) on any other model.

    Whether the chosen *model* accepts an effort level is a separate question,
    settled in the wizard and by /model — they store no level for a model
    that reports none.
    """
    effort = (config.claude.effort or "").strip().lower()
    if effort in SDK_EFFORT_LEVELS:
        return effort
    return DEFAULT_EFFORT if resolve_model(config) == DEFAULT_MODEL else None


def resolve_consolidation(config: "MemclawConfig") -> tuple[str, str | None]:
    """(model, effort) for consolidation: CLAUDE_CONSOLIDATION_MODEL /
    CLAUDE_CONSOLIDATION_EFFORT, else Opus 5.5 at medium. As in
    `resolve_effort`, an unknown level counts as unset, and an unset level
    means the default effort only on the default model (another model may
    not take an effort at all)."""
    model = (config.claude.consolidation_model or "").strip() or DEFAULT_CONSOLIDATION_MODEL
    effort = (config.claude.consolidation_effort or "").strip().lower()
    if effort in SDK_EFFORT_LEVELS:
        return model, effort
    if model == DEFAULT_CONSOLIDATION_MODEL:
        return model, DEFAULT_CONSOLIDATION_EFFORT
    return model, None


def fit_effort(levels: list[str], preferred: str) -> str | None:
    """The level to use on a model supporting *levels* when *preferred* is
    wanted: *preferred* itself, else the nearest supported level below it
    (else the lowest). None when the model takes no effort at all."""
    if not levels:
        return None
    if preferred in levels:
        return preferred
    order = list(SDK_EFFORT_LEVELS)
    rank = order.index(preferred) if preferred in order else len(order)
    below = [lv for lv in levels if lv in order and order.index(lv) < rank]
    return below[-1] if below else levels[0]


def credential(config: "MemclawConfig") -> str:
    """The credential matching `claude_auth_mode`."""
    if claude_auth_mode(config) == "subscription":
        return config.claude.oauth_token
    return config.claude.api_key


def build_env(config: "MemclawConfig") -> dict[str, str]:
    """Build the env dict for the Claude CLI subprocess.

    Scrubs every credential env var first so the subprocess never inherits
    a stale token from the parent shell, then injects exactly one credential
    based on the configured auth mode:

    - subscription → CLAUDE_CODE_OAUTH_TOKEN, billed against the Claude plan.
    - api_key      → ANTHROPIC_API_KEY, billed against Console credits.
    """
    env = {
        k: v for k, v in process_environment().items()
        if k not in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
    }
    mode = claude_auth_mode(config)
    if mode == "subscription":
        env["CLAUDE_CODE_OAUTH_TOKEN"] = config.claude.oauth_token
    elif mode == "api_key":
        env["ANTHROPIC_API_KEY"] = config.claude.api_key
    return env
