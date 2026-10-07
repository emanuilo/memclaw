"""Cursor Python SDK agent backend.

One `cursor-sdk-bridge` process (a Node subprocess) serves the whole Memclaw
process. Each chat keeps one open local agent on it, reused across turns: the
agent stores its own conversation on disk, so follow-up messages only carry
the per-turn context and the new message, and a restarted process resumes the
agent by id.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from loguru import logger

from .base import ProgressCallback, TurnResult
from .claude_models import ModelInfo
from .cursor_hooks import cursor_hooks_status, ensure_cursor_hooks
from .cursor_sdk_adapter import RunUsageTracker, collect_run_result, extract_run_text
from .mcp_bridge import HttpMcpServer, mcp_servers_for

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor

_DEFAULT_MODEL = "composer-2.5"

# Composer 2.5 default pricing (per 1M tokens) — fallback when the SDK
# doesn't return total cost on a turn-ended usage payload.
_INPUT_COST_PER_M = 0.5
_OUTPUT_COST_PER_M = 2.5

# Scrub Claude credentials when switching to Cursor so they can't shadow selection.
_DROP_KEYS = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]

# A model parameter whose id contains one of these is treated as its effort
# setting (/effort). Cursor reports parameters per model, so there is no
# fixed list of levels.
_EFFORT_PARAM_HINTS = ("effort", "reasoning")


def _cursor_api_key(config: "MemclawConfig") -> str:
    return (config.cursor_api_key or os.environ.get("CURSOR_API_KEY", "")).strip()


def _cursor_model(config: "MemclawConfig") -> str:
    model = (config.cursor_model or os.environ.get("CURSOR_MODEL", "")).strip()
    return model or _DEFAULT_MODEL


def _cursor_effort(config: "MemclawConfig") -> str | None:
    effort = (config.cursor_effort or os.environ.get("CURSOR_EFFORT", "")).strip()
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


def _build_combined_prompt(*, system_prompt: str, user_message: str) -> str:
    return "\n".join(
        [
            system_prompt.strip(),
            "",
            "---",
            "",
            user_message.strip(),
        ]
    )


def _build_user_message(
    *,
    system_prompt: str,
    user_message: str,
    context: str = "",
    image_b64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str | Any:
    """The message to send. Cursor agents have no system prompt field, so a
    new agent gets it at the top of its first message; follow-ups pass ""."""
    from cursor_sdk import SDKImage, UserMessage

    body = "\n\n".join(p for p in (context, user_message) if p)
    prompt = (
        _build_combined_prompt(system_prompt=system_prompt, user_message=body)
        if system_prompt
        else body
    )
    if not image_b64:
        return prompt
    return UserMessage(
        text=prompt,
        images=[SDKImage.from_data(image_b64, image_media_type)],
    )


def _local_agent_options(*, cwd: str) -> Any:
    from cursor_sdk import LocalAgentOptions

    return LocalAgentOptions(
        cwd=cwd,
        # Load ~/.memclaw/.cursor/hooks.json (project hooks for this cwd).
        setting_sources=["project"],
    )


def _agent_options(
    *,
    api_key: str,
    cwd: str,
    model: Any,
    mcp_servers: dict[str, Any] | None = None,
) -> Any:
    from cursor_sdk import AgentOptions

    return AgentOptions(
        api_key=api_key,
        model=model,
        local=_local_agent_options(cwd=cwd),
        mcp_servers=mcp_servers,
    )


def _effort_param(model: Any) -> Any | None:
    """The parameter of an SDKModel that acts as its effort level, if any."""
    for param in model.parameters:
        if param.values and any(h in param.id.lower() for h in _EFFORT_PARAM_HINTS):
            return param
    return None


async def _drain_stderr(stream: asyncio.StreamReader) -> None:
    """Keep reading the bridge's stderr. The SDK only reads it until the
    bridge is up, and a long-lived bridge would block once the pipe fills."""
    while line := await stream.readline():
        logger.debug("cursor-sdk-bridge: {line}", line=line.decode(errors="replace").rstrip())


def _bridge_failed(exc: BaseException) -> bool:
    """True when *exc* came from not reaching the bridge process at all
    (it crashed or hung), rather than from the agent or Cursor's API."""
    return isinstance(exc.__cause__, httpx.RequestError)


@dataclass
class _Session:
    """One chat's open Cursor agent."""

    # Serializes turns (and /new) within one chat.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent: Any = None
    # The bridge client *agent* lives on; a relaunched bridge doesn't know it.
    client: Any = None


class CursorAgentBackend:
    """Cursor Python SDK implementation of the AgentBackend protocol."""

    name: ClassVar[str] = "cursor"
    display_name: ClassVar[str] = "Cursor SDK"
    supports_model_selection: ClassVar[bool] = True

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._api_key = _cursor_api_key(config)
        self._model = _cursor_model(config)
        self._effort = _cursor_effort(config)
        self._cwd = str(config.memory_dir)
        os.environ["MEMCLAW_MEMORY_DIR"] = self._cwd
        self.bills_per_token = True
        self._mcp_server = HttpMcpServer()
        self._client: Any = None  # the shared bridge client, launched lazily
        self._client_lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()  # stderr drains
        self._sessions: dict[str, _Session] = {}
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
        memory_dir: Path | str | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        from ..setup import _masked_input

        current = existing.get("CURSOR_API_KEY", os.environ.get("CURSOR_API_KEY", ""))
        answer = _masked_input("Cursor API key (required)")
        value = answer or current
        if not value:
            console.print("[red]Error:[/red] Cursor API key is required.")
            raise SystemExit(1)

        values: dict[str, str] = {"CURSOR_API_KEY": value}
        drop_keys = list(_DROP_KEYS)

        model_current = existing.get("CURSOR_MODEL", os.environ.get("CURSOR_MODEL", ""))
        from rich.prompt import Prompt

        model_answer = Prompt.ask(
            r"Cursor model \[composer-2.5 is default, optional]",
            default="",
            show_default=False,
        )
        if model_answer.strip():
            values["CURSOR_MODEL"] = model_answer.strip()
            if model_answer.strip() != model_current:
                # An effort level picked with /effort belongs to the old model.
                drop_keys.append("CURSOR_EFFORT")
        elif model_current:
            values["CURSOR_MODEL"] = model_current

        from ..config import MemclawConfig

        cfg = MemclawConfig(memory_dir=Path(memory_dir)) if memory_dir else MemclawConfig()
        if not ensure_cursor_hooks(cfg.memory_dir):
            status = cursor_hooks_status(cfg.memory_dir)
            logger.warning(
                "Cursor tool-restriction hooks are not ready ({status}). "
                "Built-in Cursor tools may be used until hooks are installed.",
                status=status,
            )

        return values, drop_keys

    # -- Lifecycle ---------------------------------------------------------

    async def on_agent_start(self, tool_executor: "ToolExecutor") -> None:
        await self._ensure_mcp_server(tool_executor)
        if ensure_cursor_hooks(self.config.memory_dir):
            status = cursor_hooks_status(self.config.memory_dir)
            logger.info("Cursor tool-restriction hooks: {status}", status=status)
        else:
            status = cursor_hooks_status(self.config.memory_dir)
            logger.warning(
                "Cursor tool-restriction hooks are not ready ({status}). "
                "Built-in Cursor tools may be used until hooks are installed.",
                status=status,
            )

    async def on_agent_shutdown(self) -> None:
        for session in self._sessions.values():
            await self._close_agent(session)
        self._sessions.clear()
        if self._client is not None:
            await self._drop_client(self._client)
        await self._mcp_server.stop()

    async def _ensure_mcp_server(self, tool_executor: "ToolExecutor") -> None:
        await self._mcp_server.start(
            tool_executor,
            port=self.config.mcp_http_port,
        )

    # -- Bridge ------------------------------------------------------------

    async def _launch_client(self) -> Any:
        from cursor_sdk import AsyncClient
        from cursor_sdk._vendor import resolve_bridge_path

        # Local agents are stored under a path derived from the bridge
        # process's working directory (not LocalAgentOptions.cwd), and
        # launch_bridge can't set it, so start the bridge through `sh` in
        # the memory dir. A restarted process then finds its agents again.
        Path(self._cwd).mkdir(parents=True, exist_ok=True)
        client = await AsyncClient.launch_bridge(
            ["/bin/sh", "-c", 'cd "$0" && exec "$@"', self._cwd, resolve_bridge_path()],
            workspace=self._cwd,
        )
        process = getattr(client._owned_bridge, "process", None)
        if process is not None and process.stderr is not None:
            task = asyncio.create_task(_drain_stderr(process.stderr))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return client

    async def _get_client(self) -> Any:
        async with self._client_lock:
            if self._client is None:
                self._client = await self._launch_client()
            return self._client

    async def _drop_client(self, client: Any) -> None:
        """Shut *client*'s bridge down; the next call launches a new one."""
        if self._client is client:
            self._client = None
        try:
            await client.aclose()
        except Exception as exc:
            logger.warning("Error while closing the Cursor SDK bridge: {exc}", exc=exc)

    # -- Sessions ----------------------------------------------------------

    @staticmethod
    async def _close_agent(session: _Session) -> None:
        """Release the agent's executor. Its conversation stays on disk."""
        agent, session.agent, session.client = session.agent, None, None
        if agent is None:
            return
        try:
            await agent.close()
        except Exception as exc:
            logger.warning("Error while closing Cursor agent: {exc}", exc=exc)

    async def _open_agent(
        self, client: Any, *, resume: str | None, mcp_servers: dict[str, Any],
    ) -> tuple[Any, bool]:
        """Resume agent *resume*, or create a new one when there is none or it
        can't be resumed (deleted, from another machine, ...). Returns the
        agent and whether it is new."""
        from cursor_sdk import CursorAgentError

        # Local resume doesn't restore the model; pass it along explicitly.
        options = _agent_options(
            api_key=self._api_key,
            cwd=self._cwd,
            model=await self._model_selection(),
            mcp_servers=mcp_servers,
        )
        if resume:
            try:
                return await client.agents.resume(resume, options), False
            except CursorAgentError as exc:
                if _bridge_failed(exc):
                    raise
                logger.warning(
                    "Could not resume Cursor agent {id} ({exc}); starting a fresh one",
                    id=resume, exc=exc.message,
                )
        return await client.agents.create(options), True

    async def reset_session(self, session_key: str) -> None:
        session = self._sessions.get(session_key)
        if session is None:
            return
        async with session.lock:
            await self._close_agent(session)

    # -- Model selection ---------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def effort(self) -> str | None:
        return self._effort

    async def list_models(self) -> list[ModelInfo]:
        client = await self._get_client()
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

    async def _model_info(self, model_id: str) -> ModelInfo | None:
        """*model_id*'s entry in the model list, or None if it isn't listed
        or the list can't be fetched."""
        if not self._models:
            try:
                await self.list_models()
            except Exception as exc:
                logger.warning("Could not fetch the Cursor model list: {exc}", exc=exc)
                return None
        return next((m for m in self._models if m.id == model_id), None)

    async def _model_selection(self) -> Any:
        """The model (and effort parameter, if set) to run with."""
        from cursor_sdk import ModelParameterValue, ModelSelection

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
        if info is not None and self._effort and self._effort not in info.effort_levels:
            effort = None
            note = (
                f"{info.display_name} doesn't support '{self._effort}' effort, "
                "so its default is used."
            )
        self._apply_model(model_id, effort)
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

    def _apply_model(self, model: str, effort: str | None) -> None:
        """Use *model* / *effort* from the next turn on and save them to
        ~/.memclaw/.env. The model is passed with every send, so open agents
        keep their conversation and need no reconnect."""
        from ..setup import update_env_file  # local import — setup imports backends

        logger.info(
            "Cursor model: {m} effort: {e} -> {m2} effort: {e2}",
            m=self._model, e=self._effort, m2=model, e2=effort,
        )
        self._model, self._effort = model, effort
        self.config.cursor_model = model
        self.config.cursor_effort = effort or ""
        if effort:
            update_env_file({"CURSOR_MODEL": model, "CURSOR_EFFORT": effort})
        else:
            update_env_file({"CURSOR_MODEL": model}, drop_keys=["CURSOR_EFFORT"])

    # -- Runtime -----------------------------------------------------------

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        from cursor_sdk import AsyncAgent, CursorAgentError

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        prompt = _build_combined_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        options = _agent_options(
            api_key=self._api_key,
            cwd=self._cwd,
            model=await self._model_selection(),
        )

        client = await self._get_client()
        try:
            # A throwaway agent on the shared bridge (created, run, closed).
            result = await AsyncAgent.prompt(prompt, options, client=client)
        except CursorAgentError as exc:
            if _bridge_failed(exc):
                await self._drop_client(client)
            logger.error("Cursor SDK one-shot failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
        text = extract_run_text(result)
        if not text.strip():
            return "I couldn't generate a response."
        return text

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
        on_tool: ProgressCallback | None = None,
    ) -> TurnResult:
        from cursor_sdk import CursorAgentError, LocalSendOptions, SendOptions

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        await self._ensure_mcp_server(tool_executor)
        mcp_config = self._mcp_server.config
        if mcp_config is None:
            raise RuntimeError("MCP server failed to start")
        mcp_servers = mcp_servers_for(mcp_config)

        session = self._sessions.setdefault(session_key, _Session())
        async with session.lock:
            client = await self._get_client()
            if session.agent is not None and session.client is not client:
                session.agent = None  # its bridge was relaunched
            opened = is_new = False
            try:
                if session.agent is None:
                    session.agent, is_new = await self._open_agent(
                        client, resume=resume_session_id, mcp_servers=mcp_servers,
                    )
                    session.client = client
                    opened = True
                message = _build_user_message(
                    system_prompt=system_prompt if is_new else "",
                    context=context,
                    user_message=user_message,
                    image_b64=image_b64,
                    image_media_type=image_media_type,
                )
                usage_tracker = RunUsageTracker()
                run = await session.agent.send(
                    message,
                    SendOptions(
                        # Passed on every send: a resumed agent doesn't
                        # remember its model, and /model takes effect here.
                        model=await self._model_selection(),
                        mcp_servers=mcp_servers,
                        on_delta=usage_tracker.on_delta,
                        # A resumed agent may still show a run that died with
                        # the previous process; take it over.
                        local=LocalSendOptions(force=True) if opened and not is_new else None,
                    ),
                )
                result = await collect_run_result(
                    run,
                    max_turns=max_turns,
                    usage_tracker=usage_tracker,
                    on_tool=on_tool,
                )
                result.session_id = session.agent.agent_id
            except BaseException as exc:
                # The agent may be mid-run or gone; reopen (resuming) next time.
                await self._close_agent(session)
                if _bridge_failed(exc):
                    await self._drop_client(client)
                if isinstance(exc, CursorAgentError):
                    logger.error("Cursor SDK turn failed: {msg}", msg=exc.message)
                    raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
                raise

        if not result.text.strip():
            result.text = "I couldn't generate a response."
        _apply_cost_fallback(result, bills_per_token=self.bills_per_token)
        return result
