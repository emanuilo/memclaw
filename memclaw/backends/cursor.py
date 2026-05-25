"""Cursor Python SDK agent backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from .base import TurnResult
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


def _cursor_api_key(config: "MemclawConfig") -> str:
    return (config.cursor_api_key or os.environ.get("CURSOR_API_KEY", "")).strip()


def _cursor_model(config: "MemclawConfig") -> str:
    model = (config.cursor_model or os.environ.get("CURSOR_MODEL", "")).strip()
    return model or _DEFAULT_MODEL


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
    image_b64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str | Any:
    from cursor_sdk import SDKImage, UserMessage

    prompt = _build_combined_prompt(
        system_prompt=system_prompt,
        user_message=user_message,
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
    model: str,
    mcp_servers: dict[str, Any] | None = None,
) -> Any:
    from cursor_sdk import AgentOptions

    return AgentOptions(
        api_key=api_key,
        model=model,
        local=_local_agent_options(cwd=cwd),
        mcp_servers=mcp_servers,
    )


class CursorAgentBackend:
    """Cursor Python SDK implementation of the AgentBackend protocol."""

    name: ClassVar[str] = "cursor"
    display_name: ClassVar[str] = "Cursor SDK"

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._api_key = _cursor_api_key(config)
        self._model = _cursor_model(config)
        self._cwd = str(config.memory_dir)
        os.environ["MEMCLAW_MEMORY_DIR"] = self._cwd
        self.bills_per_token = True
        self._mcp_server = HttpMcpServer()

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

        model_current = existing.get("CURSOR_MODEL", os.environ.get("CURSOR_MODEL", ""))
        from rich.prompt import Prompt

        model_answer = Prompt.ask(
            r"Cursor model \[composer-2.5 is default, optional]",
            default="",
            show_default=False,
        )
        if model_answer.strip():
            values["CURSOR_MODEL"] = model_answer.strip()
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

        return values, list(_DROP_KEYS)

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
        await self._mcp_server.stop()

    async def _ensure_mcp_server(self, tool_executor: "ToolExecutor") -> None:
        await self._mcp_server.start(
            tool_executor,
            port=self.config.mcp_http_port,
        )

    async def _launch_client(self) -> Any:
        from cursor_sdk import AsyncClient

        return await AsyncClient.launch_bridge(workspace=self._cwd)

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        from contextlib import asynccontextmanager

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
            model=self._model,
        )

        @asynccontextmanager
        async def client_ctx():
            client = await self._launch_client()
            try:
                yield client
            finally:
                await client.aclose()

        try:
            async with client_ctx() as client:
                result = await AsyncAgent.prompt(prompt, options, client=client)
                text = extract_run_text(result)
                if not text.strip():
                    return "I couldn't generate a response."
                return text
        except CursorAgentError as exc:
            logger.error("Cursor SDK one-shot failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc

    async def run_turn(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tool_executor: "ToolExecutor",
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
    ) -> TurnResult:
        from cursor_sdk import CursorAgentError, SendOptions

        if not self._api_key:
            raise RuntimeError("CURSOR_API_KEY is not configured")

        await self._ensure_mcp_server(tool_executor)

        message = _build_user_message(
            system_prompt=system_prompt,
            user_message=user_message,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )

        mcp_config = self._mcp_server.config
        if mcp_config is None:
            raise RuntimeError("MCP server failed to start")

        mcp_servers = mcp_servers_for(mcp_config)
        usage_tracker = RunUsageTracker()
        client = await self._launch_client()
        try:
            agent = await client.agents.create(
                _agent_options(
                    api_key=self._api_key,
                    cwd=self._cwd,
                    model=self._model,
                    mcp_servers=mcp_servers,
                )
            )
            try:
                run = await agent.send(
                    message,
                    SendOptions(
                        mcp_servers=mcp_servers,
                        on_delta=usage_tracker.on_delta,
                    ),
                )
                result = await collect_run_result(
                    run,
                    max_turns=max_turns,
                    usage_tracker=usage_tracker,
                )
            finally:
                await agent.close()

            if not result.text.strip():
                result.text = "I couldn't generate a response."
            _apply_cost_fallback(result, bills_per_token=self.bills_per_token)
            return result
        except CursorAgentError as exc:
            logger.error("Cursor SDK turn failed: {msg}", msg=exc.message)
            raise RuntimeError(f"Cursor SDK error: {exc.message}") from exc
        finally:
            await client.aclose()
