"""Cursor Python SDK agent backend.

One `cursor-sdk-bridge` process (a Node subprocess) serves the whole Memclaw
process. Each chat keeps one open local agent on it, reused across turns: the
agent stores its own conversation on disk, so follow-up messages only carry
the per-turn context and the new message, and a restarted process resumes the
agent by id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from loguru import logger

from ...config import set_environment
from ..base import TurnRequest, TurnResult
from ..mcp_bridge import HttpMcpServer, mcp_servers_for
from ..model_selection import ModelInfo, ModelSelectionMixin
from .agents import (
    ChatAgent,
    agent_options,
    build_combined_prompt,
    build_user_message,
    prompt_once,
)
from .bridge import CursorBridge, bridge_failed
from .configure import ask_cursor_settings, install_hooks

if TYPE_CHECKING:
    from rich.console import Console

    from ...config import MemclawConfig
    from ...tools import ToolExecutor

_DEFAULT_MODEL = "composer-2.5"

# Composer 2.5 default pricing (per 1M tokens) — fallback when the SDK
# doesn't return total cost on a turn-ended usage payload.
_INPUT_COST_PER_M = 0.5
_OUTPUT_COST_PER_M = 2.5

# A model parameter whose id contains one of these is treated as its effort
# setting (/effort). Cursor reports parameters per model, so there is no
# fixed list of levels.
_EFFORT_PARAM_HINTS = ("effort", "reasoning")


def _cursor_api_key(config: "MemclawConfig") -> str:
    return config.cursor.api_key.strip()


def _cursor_model(config: "MemclawConfig") -> str:
    model = config.cursor.model.strip()
    return model or _DEFAULT_MODEL


def _cursor_effort(config: "MemclawConfig") -> str | None:
    effort = config.cursor.effort.strip()
    return effort or None


def _apply_cost_fallback(result: TurnResult, *, bills_per_token: bool) -> None:
    if result.cost_usd is not None or not bills_per_token:
        return
    if not (result.input_tokens or result.output_tokens or result.cache_read_tokens):
        return
    cache_read_cost = result.cache_read_tokens * _INPUT_COST_PER_M * 0.1 / 1_000_000
    result.cost_usd = (
        result.input_tokens * _INPUT_COST_PER_M / 1_000_000
        + result.output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        + cache_read_cost
    )


def _effort_param(model: Any) -> Any | None:
    """The parameter of an SDKModel that acts as its effort level, if any."""
    for param in model.parameters:
        if param.values and any(h in param.id.lower() for h in _EFFORT_PARAM_HINTS):
            return param
    return None


class CursorAgentBackend(ModelSelectionMixin):
    """Cursor Python SDK implementation of the AgentBackend protocol."""

    _label = "Cursor"
    _settings_group = "cursor"
    name: ClassVar[str] = "cursor"
    display_name: ClassVar[str] = "Cursor SDK"
    supports_model_selection: ClassVar[bool] = True

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._api_key = _cursor_api_key(config)
        self._model = _cursor_model(config)
        self._effort = _cursor_effort(config)
        self._cwd = str(config.memory_dir)
        # Read by the Cursor hook script, which runs in a child process.
        set_environment({"MEMCLAW_MEMORY_DIR": self._cwd})
        self.bills_per_token = True
        self._mcp_server = HttpMcpServer()
        self._bridge = CursorBridge(self._cwd)
        self._sessions: dict[str, ChatAgent] = {}
        self._models: list[ModelInfo] = []  # filled by list_models()
        self._effort_params: dict[str, str] = {}  # model id -> effort param id

    @classmethod
    def is_configured(cls, config: "MemclawConfig") -> bool:
        return bool(_cursor_api_key(config))

    @classmethod
    def configuration_help(cls) -> str:
        return (
            "Cursor SDK backend requires CURSOR_API_KEY "
            "(Cursor Dashboard → Integrations, or a team service account key).\n"
            "Optional: CURSOR_MODEL (default: composer-2.5).\n"
            "Optional: MEMCLAW_MCP_PORT (default: 17373) for the local MCP HTTP server.\n"
            "Project hooks under ~/.memclaw/.cursor/ restrict the agent to Memclaw MCP tools.\n"
            "Set AGENT_BACKEND=cursor in ~/.memclaw/.env to use this backend.\n"
            "Memclaw installs ~/.memclaw/.cursor/hooks.json to block built-in "
            "Cursor tools and allow only Memclaw MCP tools."
        )

    @classmethod
    def status_rows(cls, config: "MemclawConfig") -> list[tuple[str, str]]:
        return [
            ("Model", _cursor_model(config)),
            ("Effort", _cursor_effort(config) or "default"),
        ]

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
        *,
        config: "MemclawConfig",
    ) -> tuple[dict[str, str], list[str]]:
        return ask_cursor_settings(console, existing, config)

    # -- Lifecycle ---------------------------------------------------------

    async def on_agent_start(self, tool_executor: "ToolExecutor") -> None:
        await self._ensure_mcp_server(tool_executor)
        install_hooks(self.config.memory_dir, log_ready=True)

    async def on_agent_shutdown(self) -> None:
        # A copy: a message arriving mid-shutdown may add a session.
        for session in list(self._sessions.values()):
            await session.close()
        self._sessions.clear()
        if self._bridge.client is not None:
            await self._bridge.drop(self._bridge.client)
        await self._mcp_server.stop()

    async def _ensure_mcp_server(self, tool_executor: "ToolExecutor") -> None:
        await self._mcp_server.start(
            tool_executor,
            port=self.config.cursor.mcp_http_port,
        )

    # -- Sessions ----------------------------------------------------------

    async def _ensure_agent(
        self, session: ChatAgent, client: Any, *, resume: str | None, mcp_servers: dict[str, Any],
    ) -> Literal["open", "resumed", "created"]:
        """Give *session* an open agent on *client*: the one it already has,
        else agent *resume* resumed, else a new one. Returns which."""
        if session.is_open_on(client):
            return "open"
        # None yet, or it lived on a bridge that has since been relaunched.
        # Local resume doesn't restore the model; pass it along explicitly.
        options = agent_options(
            api_key=self._api_key,
            cwd=self._cwd,
            model=await self._model_selection(),
            mcp_servers=mcp_servers,
        )
        is_new = await session.open(client, options, resume=resume)
        return "created" if is_new else "resumed"

    async def reset_session(self, session_key: str) -> None:
        session = self._sessions.get(session_key)
        if session is None:
            return
        async with session.lock:
            await session.close()

    # -- Model selection ---------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        client = await self._bridge.get()
        models = await client.models.list(api_key=self._api_key)
        infos: list[ModelInfo] = []
        params: dict[str, str] = {}
        for m in models:
            param = _effort_param(m)
            if param is not None:
                params[m.id] = param.id
            infos.append(ModelInfo(
                id=m.id,
                display_name=m.display_name or m.id,
                created_at="",
                effort_levels=[v.value for v in param.values] if param else [],
            ))
        self._models, self._effort_params = infos, params
        return infos

    async def _model_selection(self) -> Any:
        """The model (and effort parameter, if set) to run with."""
        from cursor_sdk import ModelParameterValue, ModelSelection  # optional dependency (cursor-sdk)

        if not self._effort:
            return ModelSelection(id=self._model)
        if self._model not in self._effort_params:
            await self._model_info(self._model)  # learn the parameter's id
        param = self._effort_params.get(self._model)
        if param is None:
            return ModelSelection(id=self._model)
        return ModelSelection(
            id=self._model,
            params=[ModelParameterValue(id=param, value=self._effort)],
        )

    async def set_model(self, model_id: str) -> str:
        """Switch from the next turn on. The model is passed with every send,
        so open agents keep their conversation and need no reconnect."""
        info = await self._known_model(model_id.strip())
        effort = self._effort
        note = ""
        if self._effort and self._effort not in info.effort_levels:
            effort = None
            note = (
                f"{info.display_name} doesn't support '{self._effort}' effort, "
                "so its default is used."
            )
        self._apply_model(info.id, effort)
        return note

    async def set_effort(self, level: str) -> None:
        level = level.strip()
        levels = await self.effort_levels()
        if levels is None:
            raise ValueError("Couldn't load the model list right now; try again later.")
        if not levels:
            raise ValueError("The current model doesn't support an effort setting.")
        match = next((v for v in levels if v.lower() == level.lower()), None)
        if match is None:
            raise ValueError(
                f"The current model doesn't support '{level}'. "
                f"Supported: {', '.join(levels)}."
            )
        self._apply_model(self._model, match)

    # -- Runtime -----------------------------------------------------------

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        from cursor_sdk import CursorAgentError  # optional dependency (cursor-sdk)

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        prompt = build_combined_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        options = agent_options(
            api_key=self._api_key,
            cwd=self._cwd,
            model=await self._model_selection(),
        )

        client = await self._bridge.get()
        try:
            text = await prompt_once(prompt, options, client)
        except CursorAgentError as exc:
            if bridge_failed(exc):
                await self._bridge.drop(client)
            logger.error("Cursor SDK one-shot failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
        if not text.strip():
            # Consolidation would write a placeholder reply over MEMORY.md.
            raise RuntimeError("Cursor SDK one-shot returned no text")
        return text

    async def run_turn(self, request: TurnRequest, tool_executor: "ToolExecutor") -> TurnResult:
        from cursor_sdk import CursorAgentError  # optional dependency (cursor-sdk)

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        await self._ensure_mcp_server(tool_executor)
        mcp_config = self._mcp_server.config
        if mcp_config is None:
            raise RuntimeError("MCP server failed to start")
        mcp_servers = mcp_servers_for(mcp_config)

        session = self._sessions.setdefault(request.session_key, ChatAgent())
        async with session.lock:
            client = await self._bridge.get()
            try:
                agent = await self._ensure_agent(
                    session, client, resume=request.resume_session_id, mcp_servers=mcp_servers,
                )
                message = build_user_message(
                    # A Cursor agent has no system-prompt field; a new one
                    # gets it at the top of its first message.
                    system_prompt=request.system_prompt if agent == "created" else "",
                    context=request.context,
                    user_message=request.user_message,
                    image_b64=request.image_b64,
                    image_media_type=request.image_media_type,
                )
                result = await session.send(
                    message,
                    # Passed on every send: a resumed agent doesn't remember
                    # its model, and /model takes effect here.
                    model=await self._model_selection(),
                    mcp_servers=mcp_servers,
                    # A resumed agent may still show a run that died with
                    # the previous process; take it over.
                    force=agent == "resumed",
                    max_turns=request.max_turns, on_tool=request.on_tool,
                )
            except BaseException as exc:
                # The agent may be mid-run or gone; reopen (resuming) next time.
                await session.close()
                if bridge_failed(exc):
                    await self._bridge.drop(client)
                if isinstance(exc, CursorAgentError):
                    logger.error("Cursor SDK turn failed: {msg}", msg=exc.message)
                    raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
                raise

        if not result.text.strip():
            result.text = "I couldn't generate a response."
        _apply_cost_fallback(result, bills_per_token=self.bills_per_token)
        return result
