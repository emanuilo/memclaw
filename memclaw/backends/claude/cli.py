"""Running the Claude CLI: the locked-down options every call shares, one
connected `ClaudeSDKClient` (one CLI process) per chat, reused across
turns, and one-shot calls.

Keeping the client connected means the conversation lives in the CLI's own
session instead of being re-sent with every message.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)
from loguru import logger

from ...tools import TOOL_DEFINITIONS
from ..base import TurnRequest, TurnResult
from ..mcp_tools import MCP_SERVER_NAME
from .cost import SessionCosts
from .models import sdk_effort
from .replies import compose_prompt, read_one_shot, read_turn
from .settings import build_env

if TYPE_CHECKING:
    from ...config import MemclawConfig
    from ...tools import ToolExecutor

ALLOWED_TOOLS = [
    f"mcp__{MCP_SERVER_NAME}__{t['name']}" for t in TOOL_DEFINITIONS
]

# The (model, effort) to run with, read when a turn starts.
ModelChoice = Callable[[], "tuple[str, str | None]"]


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


@dataclass
class _Session:
    """One chat's live Claude CLI connection."""

    # Serializes turns (and /new) within one chat.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: ClaudeSDKClient | None = None
    # (model, effort) the client was started with. After /model or /effort
    # the next turn reconnects, resuming the same session with the new ones.
    options_key: tuple[str, str | None] | None = None


class ClaudeCli:
    """The Claude CLI processes of one backend: a connection per chat, and
    the cost of each chat's turns."""

    def __init__(self, config: "MemclawConfig", *, bills_per_token: bool) -> None:
        self._env = build_env(config)
        self._cwd = str(config.memory_dir)
        self._mcp_server: Any = None  # built lazily, needs a ToolExecutor
        self._sessions: dict[str, _Session] = {}
        self.costs = SessionCosts(bills_per_token=bills_per_token)

    def options(
        self, *, model: str, effort: str | None, system_prompt: str, max_turns: int, **extra: Any,
    ) -> ClaudeAgentOptions:
        """The locked-down CLI options every call shares (chat sessions and
        consolidation alike), plus *extra*."""
        return ClaudeAgentOptions(
            env=self._env,
            model=model,
            effort=sdk_effort(effort),
            system_prompt=system_prompt,
            # Don't load ~/.claude settings, plugins, CLAUDE.md or the user's
            # MCP servers: faster startup and nothing outside our control.
            setting_sources=[],
            strict_mcp_config=True,
            # No built-in tools at all; only our in-process mcp__memclaw__*.
            tools=[],
            # Anything not in allowed_tools is denied, never prompted for.
            permission_mode="dontAsk",
            # No @file expansion / slash commands from chat text.
            verbatim_prompts=True,
            max_turns=max_turns,
            stderr=lambda line: logger.debug("claude-cli: {line}", line=line.rstrip()),
            **extra,
        )

    # -- One-shot calls ----------------------------------------------------

    async def ask_once(
        self, *, model: str, effort: str | None, system_prompt: str, user_message: str,
    ) -> str:
        """Answer *user_message* in a throwaway CLI with no tools."""
        options = self.options(model=model, effort=effort, system_prompt=system_prompt, max_turns=1)
        async with ClaudeSDKClient(options=options) as client:
            reply = await read_one_shot(client, user_message)
        return reply.answer()

    # -- Chat sessions -----------------------------------------------------

    def _session_options(
        self, choice: ModelChoice, *, system_prompt: str, resume: str | None, max_turns: int,
    ) -> ClaudeAgentOptions:
        model, effort = choice()
        return self.options(
            model=model,
            effort=effort,
            system_prompt=system_prompt,
            max_turns=max_turns,
            # Claude Code stores sessions per working directory; a fixed one
            # lets a restarted process find (and resume) them.
            cwd=self._cwd,
            mcp_servers={MCP_SERVER_NAME: self._mcp_server},
            allowed_tools=ALLOWED_TOOLS,
            resume=resume,
        )

    async def _connect(
        self, choice: ModelChoice, *, system_prompt: str, resume: str | None, max_turns: int,
    ) -> ClaudeSDKClient:
        """Start a CLI for one chat, resuming *resume* when given. A session
        that can't be resumed (deleted, from another machine, ...) is
        replaced by a fresh one rather than failing the turn."""
        if resume:
            client = ClaudeSDKClient(options=self._session_options(
                choice, system_prompt=system_prompt, resume=resume, max_turns=max_turns,
            ))
            try:
                await client.connect()
                return client
            except Exception as exc:
                logger.exception(
                    "Could not resume Claude session {id} ({exc}); starting a fresh one",
                    id=resume, exc=exc,
                )
        client = ClaudeSDKClient(options=self._session_options(
            choice, system_prompt=system_prompt, resume=None, max_turns=max_turns,
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
            logger.exception("Error while disconnecting Claude CLI: {exc}", exc=exc)

    async def reset(self, session_key: str) -> None:
        session = self._sessions.get(session_key)
        if session is None:
            return
        async with session.lock:
            await self._disconnect(session)

    async def close_all(self) -> None:
        for session in self._sessions.values():
            await self._disconnect(session)
        self._sessions.clear()

    async def run_turn(
        self, request: TurnRequest, tool_executor: "ToolExecutor", choice: ModelChoice,
    ) -> TurnResult:
        """Run *request* on its chat's CLI, connecting (or reconnecting with
        the model and effort *choice* gives now) first if needed."""
        if self._mcp_server is None:
            self._mcp_server = _build_mcp_server(tool_executor)

        session = self._sessions.setdefault(request.session_key, _Session())
        async with session.lock:
            options_key = choice()
            if session.client is not None and session.options_key != options_key:
                await self._disconnect(session)
            if session.client is None:
                session.client = await self._connect(
                    choice,
                    system_prompt=request.system_prompt,
                    resume=request.resume_session_id,
                    max_turns=request.max_turns,
                )
                session.options_key = options_key
            try:
                return await read_turn(
                    session.client,
                    compose_prompt(request.context, request.user_message),
                    image_b64=request.image_b64,
                    image_media_type=request.image_media_type,
                    on_tool=request.on_tool,
                )
            except BaseException:
                # The CLI may be dead or mid-turn; reconnect (resuming) next time.
                await self._disconnect(session)
                raise
