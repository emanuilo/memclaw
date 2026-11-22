"""Claude Agent SDK backend for Memclaw.

Wraps `claude-agent-sdk` (which itself shells out to the Claude CLI) into the
neutral `AgentBackend` protocol: the settings and credentials, /model and
/effort, and the turns, which run on each chat's own CLI process (`cli`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ..base import TurnRequest, TurnResult
from ..model_selection import ModelInfo, ModelSelectionMixin
from .cli import ClaudeCli
from .configure import ask_claude_settings
from .models import SDK_EFFORT_LEVELS, fetch_models
from .settings import (
    DEFAULT_EFFORT,
    claude_auth_mode,
    credential,
    fit_effort,
    resolve_consolidation,
    resolve_effort,
    resolve_model,
)

if TYPE_CHECKING:
    from rich.console import Console

    from ...config import MemclawConfig
    from ...tools import ToolExecutor


class ClaudeAgentBackend(ModelSelectionMixin):
    """Claude Agent SDK implementation of the AgentBackend protocol."""

    _label = "Claude"
    _settings_group = "claude"
    name: ClassVar[str] = "claude"
    display_name: ClassVar[str] = "Claude Agent SDK"
    supports_model_selection: ClassVar[bool] = True

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._model = resolve_model(config)
        self._effort = resolve_effort(config)
        self._models: list[ModelInfo] = []  # filled by list_models()
        # Subscription billing → no per-message dollar figure in logs.
        self.bills_per_token = claude_auth_mode(config) == "api_key"
        self._cli = ClaudeCli(config, bills_per_token=self.bills_per_token)

    # -- Configuration ---------------------------------------------------

    @classmethod
    def is_configured(cls, config: "MemclawConfig") -> bool:
        return bool(claude_auth_mode(config))

    @classmethod
    def configuration_help(cls) -> str:
        return (
            "no Claude credential is configured.\n"
            "Choose one:\n"
            "  • Claude subscription — generate a token with `claude setup-token` "
            "and save it as CLAUDE_CODE_OAUTH_TOKEN.\n"
            "  • Anthropic API key — set ANTHROPIC_API_KEY (billed per token)."
        )

    @classmethod
    def status_rows(cls, config: "MemclawConfig") -> list[tuple[str, str]]:
        consolidation_model, consolidation_effort = resolve_consolidation(config)
        return [
            ("Model", resolve_model(config)),
            ("Effort", resolve_effort(config) or "default"),
            ("Consolidation", f"{consolidation_model} ({consolidation_effort or 'default'})"),
        ]

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
        *,
        config: "MemclawConfig",
    ) -> tuple[dict[str, str], list[str]]:
        """Ask whether to use a subscription or API key, then prompt the
        chosen credential. Returns (values_to_save, env_keys_to_drop).
        """
        return ask_claude_settings(console, existing, config)

    # -- Lifecycle ---------------------------------------------------------

    async def on_agent_start(self, tool_executor: "ToolExecutor") -> None:
        pass

    async def on_agent_shutdown(self) -> None:
        await self._cli.close_all()

    async def reset_session(self, session_key: str) -> None:
        await self._cli.reset(session_key)

    # -- Model selection ---------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        self._models = await fetch_models(claude_auth_mode(self.config), credential(self.config))
        return self._models

    async def set_model(self, model_id: str) -> str:
        """Switch from the next turn on. Live chats then reconnect, resuming
        their session (Claude Code resumes fine under a different model)."""
        info = await self._known_model(model_id.strip())
        # Keep the effort if the new model takes it, otherwise the nearest
        # level it does take (or none at all).
        effort = fit_effort(info.effort_levels, self._effort or DEFAULT_EFFORT)
        note = ""
        if self._effort and effort is None:
            note = f"{info.display_name} doesn't support an effort setting, so none is used."
        elif self._effort and effort != self._effort:
            note = (
                f"{info.display_name} doesn't support '{self._effort}' effort; "
                f"using '{effort}' instead."
            )
        self._apply_model(info.id, effort)
        return note

    async def set_effort(self, level: str) -> None:
        level = level.strip().lower()
        if level not in SDK_EFFORT_LEVELS:
            raise ValueError(
                f"Unknown effort '{level}'. Levels: {', '.join(SDK_EFFORT_LEVELS)}."
            )
        levels = await self.effort_levels()
        if levels == []:
            raise ValueError("The current model doesn't support an effort setting.")
        if levels is not None and level not in levels:
            raise ValueError(
                f"The current model doesn't support '{level}'. "
                f"Supported: {', '.join(levels)}."
            )
        self._apply_model(self._model, level)

    # -- Runtime ---------------------------------------------------------

    async def run_one_shot(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        # Only memory consolidation uses one-shot calls.
        model, effort = resolve_consolidation(self.config)
        return await self._cli.ask_once(
            model=model, effort=effort, system_prompt=system_prompt, user_message=user_message,
        )

    async def run_turn(self, request: TurnRequest, tool_executor: "ToolExecutor") -> TurnResult:
        result = await self._cli.run_turn(
            request, tool_executor, lambda: (self._model, self._effort),
        )
        result.cost_usd = self._cli.costs.turn_cost(
            result, request.resume_session_id, model=self._model,
        )
        return result
