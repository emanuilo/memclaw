"""Claude Agent SDK backend for Memclaw.

Wraps `claude-agent-sdk` (which itself shells out to the Claude CLI) into the
neutral `AgentBackend` protocol. All Claude-specific glue — env scrubbing,
MCP server wrapping for tools, stream-json image protocol, response parsing,
per-chat CLI sessions — lives in this file.

Each chat keeps one connected `ClaudeSDKClient` (one Claude CLI process)
that is reused across turns, so the conversation lives in the CLI's own
session instead of being re-sent with every message.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from loguru import logger
from ..prompts import choose
from ..tools import TOOL_DEFINITIONS
from .base import TurnResult
from .claude_models import SDK_EFFORT_LEVELS, ModelInfo, fetch_models
from .mcp_tools import MCP_SERVER_NAME

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor


_DEFAULT_MODEL = "claude-sonnet-5-5"

# Used when CLAUDE_EFFORT is unset and the default model runs, and
# preselected in the wizard for any model that supports effort at all.
_DEFAULT_EFFORT = "medium"

# Anthropic list prices, (input, output) USD per 1M tokens. Only used as a
# fallback when the SDK doesn't return total_cost_usd for an API-key turn.
# Ids are matched by prefix, so dated snapshots (claude-haiku-4-5-20251001)
# resolve too. A model missing here (an alias, or one released later) is
# estimated at the default model's price, so the figure is approximate.
_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Cache reads are billed at a tenth of the input price.
_CACHE_READ_MULTIPLIER = 0.1

_ALLOWED_TOOLS = [
    f"mcp__{MCP_SERVER_NAME}__{t['name']}" for t in TOOL_DEFINITIONS
]


# ── Env / auth helpers ──────────────────────────────────────────────

def _claude_auth_mode(config: "MemclawConfig") -> str:
    """Which Claude credential is configured.

    Returns "subscription" when CLAUDE_CODE_OAUTH_TOKEN is set (billed
    against the Claude plan, no per-message cost), "api_key" when only
    ANTHROPIC_API_KEY is set, or "" when neither is set.
    """
    if config.claude_code_oauth_token:
        return "subscription"
    if config.anthropic_api_key:
        return "api_key"
    return ""


def _resolve_model(config: "MemclawConfig") -> str:
    """The configured model, or the built-in default when unset."""
    return (config.claude_model or "").strip() or _DEFAULT_MODEL


def _prices_per_m(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for *model*; see _PRICES_PER_M."""
    # Longest prefix first, so claude-opus-5-5 doesn't match claude-opus-5.
    for prefix in sorted(_PRICES_PER_M, key=len, reverse=True):
        if model == prefix or model.startswith(f"{prefix}-"):
            return _PRICES_PER_M[prefix]
    return _PRICES_PER_M[_DEFAULT_MODEL]


def _resolve_effort(config: "MemclawConfig") -> str | None:
    """The configured effort level, or None to let the model decide.

    Anything the SDK does not recognise is treated as unset: ~/.memclaw/.env
    is a plain text file, and a typo there should not reach the CLI as an
    argument it will reject. An unset level means `_DEFAULT_EFFORT` on the
    default model and None (the SDK omits the flag) on any other model.

    Whether the chosen *model* accepts an effort level is a separate question,
    settled in the wizard and by /model — they store no level for a model
    that reports none.
    """
    effort = (config.claude_effort or "").strip().lower()
    if effort in SDK_EFFORT_LEVELS:
        return effort
    return _DEFAULT_EFFORT if _resolve_model(config) == _DEFAULT_MODEL else None


def _fit_effort(levels: list[str], preferred: str) -> str | None:
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


def _credential(config: "MemclawConfig") -> str:
    """The credential matching `_claude_auth_mode`."""
    if _claude_auth_mode(config) == "subscription":
        return config.claude_code_oauth_token
    return config.anthropic_api_key


def _build_env(config: "MemclawConfig") -> dict[str, str]:
    """Build the env dict for the Claude CLI subprocess.

    Scrubs every credential env var first so the subprocess never inherits
    a stale token from the parent shell, then injects exactly one credential
    based on the configured auth mode:

    - subscription → CLAUDE_CODE_OAUTH_TOKEN, billed against the Claude plan.
    - api_key      → ANTHROPIC_API_KEY, billed against Console credits.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
    }
    mode = _claude_auth_mode(config)
    if mode == "subscription":
        env["CLAUDE_CODE_OAUTH_TOKEN"] = config.claude_code_oauth_token
    elif mode == "api_key":
        env["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    return env


# ── MCP server (tools) ──────────────────────────────────────────────

def _build_mcp_server(executor: "ToolExecutor"):
    """Wrap each TOOL_DEFINITIONS entry as an @tool-decorated async function
    bound to *executor*, and bundle them into an in-process SDK MCP server.

    Claude sees these as `mcp__memclaw__<name>`.
    """

    def _make_wrapper(tool_name: str):
        async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            result_text = await executor.execute(tool_name, args)
            return {"content": [{"type": "text", "text": result_text}]}
        wrapper.__name__ = f"tool_{tool_name}"
        return wrapper

    sdk_tools = []
    for defn in TOOL_DEFINITIONS:
        wrapped = tool(
            name=defn["name"],
            description=defn["description"],
            input_schema=defn["input_schema"],
        )(_make_wrapper(defn["name"]))
        sdk_tools.append(wrapped)

    return create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=sdk_tools)


# ── Image-input streaming protocol ──────────────────────────────────

def _compose_prompt(context: str, user_message: str) -> str:
    """The per-turn context block goes at the top of the user message, so
    the session's system prompt stays static and cacheable."""
    return f"{context}\n\n{user_message}" if context else user_message


async def _image_prompt_stream(
    message: str, image_b64: str, image_media_type: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield a single streaming-input user message containing an image + text.

    The Claude CLI's stream-json protocol expects Anthropic-style content
    blocks here, so we pass an "image" block with a base64 source followed
    by the user's text. Targets claude-agent-sdk 0.2.x.
    """
    yield {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": message},
            ],
        },
        "parent_tool_use_id": None,
    }


# ── Backend ─────────────────────────────────────────────────────────

@dataclass
class _Session:
    """One chat's live Claude CLI connection."""

    # Serializes turns (and /new) within one chat.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: ClaudeSDKClient | None = None
    # (model, effort) the client was started with. After /model or /effort
    # the next turn reconnects, resuming the same session with the new ones.
    options_key: tuple[str, str | None] | None = None


class ClaudeAgentBackend:
    """Claude Agent SDK implementation of the AgentBackend protocol."""

    name: ClassVar[str] = "claude"
    display_name: ClassVar[str] = "Claude Agent SDK"
    supports_model_selection: ClassVar[bool] = True

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._env = _build_env(config)
        self._model = _resolve_model(config)
        self._effort = _resolve_effort(config)
        self._mcp_server: Any = None  # built lazily, needs a ToolExecutor
        self._sessions: dict[str, _Session] = {}
        self._models: list[ModelInfo] = []  # filled by list_models()
        # Last total_cost_usd seen per session id. The CLI reports a running
        # total for the session (carried over on resume), not a per-turn cost.
        self._session_costs: dict[str, float] = {}
        # Subscription billing → no per-message dollar figure in logs.
        self.bills_per_token = _claude_auth_mode(config) == "api_key"

    # -- Configuration ---------------------------------------------------

    @classmethod
    def is_configured(cls, config: "MemclawConfig") -> bool:
        return bool(_claude_auth_mode(config))

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
        return [
            ("Model", _resolve_model(config)),
            ("Effort", _resolve_effort(config) or "default"),
        ]

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
        *,
        memory_dir: Path | str | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        """Ask whether to use a subscription or API key, then prompt the
        chosen credential. Returns (values_to_save, env_keys_to_drop).
        """
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

        if choice == 0:
            auth_mode = "subscription"
            env_key = "CLAUDE_CODE_OAUTH_TOKEN"
            label = "Claude subscription OAuth token (run `claude setup-token`)"
            drop_key = "ANTHROPIC_API_KEY"
        else:
            auth_mode = "api_key"
            env_key = "ANTHROPIC_API_KEY"
            label = "Anthropic API key (sk-ant-...)"
            drop_key = "CLAUDE_CODE_OAUTH_TOKEN"

        current = existing.get(env_key, "")
        from ..setup import _masked_input  # local import — setup imports backends
        answer = _masked_input(f"{label} (required)")
        value = answer or current
        if not value:
            console.print(f"[red]Error:[/red] {label} is required.")
            raise SystemExit(1)

        values = {env_key: value}
        drop_keys = [drop_key]

        model_values, model_drops = cls._ask_model_and_effort(
            console, existing, auth_mode=auth_mode, credential=value,
        )
        values.update(model_values)
        drop_keys.extend(model_drops)
        return values, drop_keys

    @classmethod
    def _ask_model_and_effort(
        cls,
        console: "Console",
        existing: dict[str, str],
        *,
        auth_mode: str,
        credential: str,
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
        current_model = (
            existing.get("CLAUDE_MODEL") or os.environ.get("CLAUDE_MODEL", "")
        ).strip() or _DEFAULT_MODEL
        current_effort = (
            existing.get("CLAUDE_EFFORT") or os.environ.get("CLAUDE_EFFORT", "")
        ).strip().lower()

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

        picked = cls._pick_model(console, models, current_model)
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

        values["CLAUDE_EFFORT"] = cls._pick_effort(
            console, picked.effort_levels, current_effort,
        )
        return values, []

    @staticmethod
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

    @staticmethod
    def _pick_effort(
        console: "Console", levels: list[str], current_effort: str,
    ) -> str:
        """Show the effort picker for a model that supports one.

        *levels* only ever holds levels this model accepts, so the preselected
        value falls back to the first one when the model has no `medium`.
        """
        preferred = current_effort if current_effort in levels else _DEFAULT_EFFORT
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

    # -- Lifecycle ---------------------------------------------------------

    async def on_agent_start(self, tool_executor: "ToolExecutor") -> None:
        pass

    async def on_agent_shutdown(self) -> None:
        for session in self._sessions.values():
            await self._disconnect(session)
        self._sessions.clear()

    # -- Sessions ----------------------------------------------------------

    def _session_options(
        self, *, system_prompt: str, resume: str | None, max_turns: int,
    ) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            env=self._env,
            model=self._model,
            effort=self._effort,
            system_prompt=system_prompt,
            # Claude Code stores sessions per working directory; a fixed one
            # lets a restarted process find (and resume) them.
            cwd=str(self.config.memory_dir),
            # Don't load ~/.claude settings, plugins, CLAUDE.md or the user's
            # MCP servers: faster startup and nothing outside our control.
            setting_sources=[],
            strict_mcp_config=True,
            mcp_servers={MCP_SERVER_NAME: self._mcp_server},
            # No built-in tools at all; only our in-process mcp__memclaw__*.
            tools=[],
            allowed_tools=_ALLOWED_TOOLS,
            # Anything not in allowed_tools is denied, never prompted for.
            permission_mode="dontAsk",
            # No @file expansion / slash commands from chat text.
            verbatim_prompts=True,
            max_turns=max_turns,
            resume=resume,
            stderr=lambda line: logger.debug("claude-cli: {line}", line=line.rstrip()),
        )

    async def _connect(
        self, *, system_prompt: str, resume: str | None, max_turns: int,
    ) -> ClaudeSDKClient:
        """Start a CLI for one chat, resuming *resume* when given. A session
        that can't be resumed (deleted, from another machine, ...) is
        replaced by a fresh one rather than failing the turn."""
        if resume:
            client = ClaudeSDKClient(options=self._session_options(
                system_prompt=system_prompt, resume=resume, max_turns=max_turns,
            ))
            try:
                await client.connect()
                return client
            except Exception as exc:
                logger.warning(
                    "Could not resume Claude session {id} ({exc}); starting a fresh one",
                    id=resume, exc=exc,
                )
        client = ClaudeSDKClient(options=self._session_options(
            system_prompt=system_prompt, resume=None, max_turns=max_turns,
        ))
        await client.connect()
        return client

    @staticmethod
    async def _disconnect(session: _Session) -> None:
        client, session.client = session.client, None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception as exc:
            logger.warning("Error while disconnecting Claude CLI: {exc}", exc=exc)

    async def reset_session(self, session_key: str) -> None:
        session = self._sessions.get(session_key)
        if session is None:
            return
        async with session.lock:
            await self._disconnect(session)

    # -- Model selection ---------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def effort(self) -> str | None:
        return self._effort

    async def list_models(self) -> list[ModelInfo]:
        self._models = await fetch_models(_claude_auth_mode(self.config), _credential(self.config))
        return self._models

    async def _model_info(self, model_id: str) -> ModelInfo | None:
        """*model_id*'s entry in the model list, or None if it isn't listed
        or the list can't be fetched."""
        if not self._models:
            try:
                await self.list_models()
            except Exception as exc:
                logger.warning("Could not fetch the Claude model list: {exc}", exc=exc)
                return None
        return next((m for m in self._models if m.id == model_id), None)

    async def effort_levels(self) -> list[str] | None:
        info = await self._model_info(self._model)
        return list(info.effort_levels) if info else None

    async def set_model(self, model_id: str) -> str:
        model_id = model_id.strip()
        info = await self._model_info(model_id)
        if info is None and self._models:
            raise ValueError(f"Unknown model '{model_id}'.")
        effort = self._effort
        note = ""
        if info is not None:
            # Keep the effort if the new model takes it, otherwise the nearest
            # level it does take (or none at all).
            effort = _fit_effort(info.effort_levels, self._effort or _DEFAULT_EFFORT)
            if self._effort and effort is None:
                note = f"{info.display_name} doesn't support an effort setting, so none is used."
            elif self._effort and effort != self._effort:
                note = (
                    f"{info.display_name} doesn't support '{self._effort}' effort; "
                    f"using '{effort}' instead."
                )
        self._apply_model(model_id, effort)
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

    def _apply_model(self, model: str, effort: str | None) -> None:
        """Use *model* / *effort* from the next turn on and save them to
        ~/.memclaw/.env. Live chats reconnect on their next turn, resuming
        their session (Claude Code resumes fine under a different model)."""
        from ..setup import update_env_file  # local import — setup imports backends

        logger.info(
            "Claude model: {m} effort: {e} -> {m2} effort: {e2}",
            m=self._model, e=self._effort, m2=model, e2=effort,
        )
        self._model, self._effort = model, effort
        self.config.claude_model = model
        self.config.claude_effort = effort or ""
        if effort:
            update_env_file({"CLAUDE_MODEL": model, "CLAUDE_EFFORT": effort})
        else:
            update_env_file({"CLAUDE_MODEL": model}, drop_keys=["CLAUDE_EFFORT"])

    # -- Runtime ---------------------------------------------------------

    async def run_one_shot(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        options = ClaudeAgentOptions(
            env=self._env,
            model=self._model,
            effort=self._effort,
            system_prompt=system_prompt,
            setting_sources=[],
            strict_mcp_config=True,
            tools=[],
            verbatim_prompts=True,
            max_turns=1,
        )

        result_text = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_message)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text
                elif isinstance(msg, ResultMessage):
                    if msg.result and not result_text:
                        result_text = msg.result
        return result_text

    async def run_turn(
        self,
        *,
        system_prompt: str,
        context: str,
        user_message: str,
        tool_executor: "ToolExecutor",
        session_key: str,
        resume_session_id: str | None = None,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
    ) -> TurnResult:
        if self._mcp_server is None:
            self._mcp_server = _build_mcp_server(tool_executor)

        session = self._sessions.setdefault(session_key, _Session())
        async with session.lock:
            options_key = (self._model, self._effort)
            if session.client is not None and session.options_key != options_key:
                await self._disconnect(session)
            if session.client is None:
                session.client = await self._connect(
                    system_prompt=system_prompt,
                    resume=resume_session_id,
                    max_turns=max_turns,
                )
                session.options_key = options_key
            try:
                result = await self._query(
                    session.client,
                    _compose_prompt(context, user_message),
                    image_b64=image_b64,
                    image_media_type=image_media_type,
                )
            except BaseException:
                # The CLI may be dead or mid-turn; reconnect (resuming) next time.
                await self._disconnect(session)
                raise

        # Turn the session's running total into this turn's cost. Without a
        # baseline (first turn after a restart resumed it) estimate instead.
        total = result.cost_usd
        result.cost_usd = None
        if total is not None and result.session_id:
            previous = self._session_costs.get(result.session_id)
            self._session_costs[result.session_id] = total
            if previous is not None or resume_session_id != result.session_id:
                result.cost_usd = max(total - (previous or 0.0), 0.0)

        # Fallback cost estimate when the SDK didn't supply a usable one.
        if result.cost_usd is None and self.bills_per_token:
            input_per_m, output_per_m = _prices_per_m(self._model)
            result.cost_usd = (
                result.input_tokens * input_per_m
                + result.output_tokens * output_per_m
                + result.cache_read_tokens * input_per_m * _CACHE_READ_MULTIPLIER
            ) / 1_000_000
        return result

    @staticmethod
    async def _query(
        client: ClaudeSDKClient,
        prompt: str,
        *,
        image_b64: str | None,
        image_media_type: str,
    ) -> TurnResult:
        """Send one user message on a connected client and collect the reply."""
        if image_b64:
            await client.query(_image_prompt_stream(prompt, image_b64, image_media_type))
        else:
            await client.query(prompt)

        last_text = ""
        result = TurnResult(text="")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                turn_text = ""
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        turn_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        args_str = json.dumps(block.input, ensure_ascii=False)
                        if len(args_str) > 300:
                            args_str = args_str[:300] + "..."
                        tool_name = block.name
                        prefix = f"mcp__{MCP_SERVER_NAME}__"
                        if tool_name.startswith(prefix):
                            tool_name = tool_name[len(prefix):]
                        logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)
                if turn_text:
                    last_text = turn_text
            elif isinstance(msg, ResultMessage):
                result.session_id = msg.session_id or None
                result.num_turns = msg.num_turns
                result.cost_usd = msg.total_cost_usd
                if msg.usage:
                    result.input_tokens = msg.usage.get("input_tokens", 0) or 0
                    result.output_tokens = msg.usage.get("output_tokens", 0) or 0
                    result.cache_read_tokens = msg.usage.get("cache_read_input_tokens", 0) or 0
                    result.cache_creation_tokens = msg.usage.get("cache_creation_input_tokens", 0) or 0
                if msg.result and not last_text:
                    last_text = msg.result

        result.text = last_text
        return result
