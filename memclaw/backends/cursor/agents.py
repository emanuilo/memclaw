"""Talking to Cursor agents: one chat's agent (`ChatAgent`), the options
and messages it is opened and sent with, and one-shot prompts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..base import ProgressCallback, TurnResult
from .bridge import bridge_failed
from .sdk_adapter import RunUsageTracker, collect_run_result, extract_run_text


def build_combined_prompt(*, system_prompt: str, user_message: str) -> str:
    return "\n".join(
        [
            system_prompt.strip(),
            "",
            "---",
            "",
            user_message.strip(),
        ]
    )


def build_user_message(
    *,
    system_prompt: str,
    user_message: str,
    context: str = "",
    image_b64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str | Any:
    """The message to send. Cursor agents have no system prompt field, so a
    new agent gets it at the top of its first message; follow-ups pass ""."""
    from cursor_sdk import SDKImage, UserMessage  # optional dependency (cursor-sdk)

    body = "\n\n".join(p for p in (context, user_message) if p)
    prompt = (
        build_combined_prompt(system_prompt=system_prompt, user_message=body)
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
    from cursor_sdk import LocalAgentOptions  # optional dependency (cursor-sdk)

    return LocalAgentOptions(
        cwd=cwd,
        # Load ~/.memclaw/.cursor/hooks.json (project hooks for this cwd).
        setting_sources=["project"],
    )


def agent_options(
    *,
    api_key: str,
    cwd: str,
    model: Any,
    mcp_servers: dict[str, Any] | None = None,
) -> Any:
    from cursor_sdk import AgentOptions  # optional dependency (cursor-sdk)

    return AgentOptions(
        api_key=api_key,
        model=model,
        local=_local_agent_options(cwd=cwd),
        mcp_servers=mcp_servers,
    )


async def prompt_once(prompt: str, options: Any, client: Any) -> str:
    """Run *prompt* on a throwaway agent on the shared bridge (created, run,
    closed) and return its reply text."""
    from cursor_sdk import AsyncAgent  # optional dependency (cursor-sdk)

    result = await AsyncAgent.prompt(prompt, options, client=client)
    return extract_run_text(result)


@dataclass
class ChatAgent:
    """One chat's open Cursor agent."""

    # Serializes turns (and /new) within one chat.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent: Any = None
    # The bridge client *agent* lives on; a relaunched bridge doesn't know it.
    client: Any = None

    def is_open_on(self, client: Any) -> bool:
        return self.agent is not None and self.client is client

    async def open(self, client: Any, options: Any, *, resume: str | None) -> bool:
        """Resume agent *resume* on *client*, or create a new one when there
        is none or it can't be resumed (deleted, from another machine, ...).
        Returns whether the agent is new."""
        from cursor_sdk import CursorAgentError  # optional dependency (cursor-sdk)

        if resume:
            try:
                agent = await client.agents.resume(resume, options)
            except CursorAgentError as exc:
                if bridge_failed(exc):
                    raise
                logger.warning(
                    "Could not resume Cursor agent {id} ({exc}); starting a fresh one",
                    id=resume, exc=exc.message,
                )
            else:
                self.agent, self.client = agent, client
                return False
        self.agent = await client.agents.create(options)
        self.client = client
        return True

    async def send(
        self,
        message: Any,
        *,
        model: Any,
        mcp_servers: dict[str, Any],
        force: bool,
        max_turns: int,
        on_tool: ProgressCallback | None,
    ) -> TurnResult:
        """Send *message* on the open agent and collect the run's reply."""
        from cursor_sdk import LocalSendOptions, SendOptions  # optional dependency (cursor-sdk)

        usage_tracker = RunUsageTracker()
        run = await self.agent.send(
            message,
            SendOptions(
                model=model,
                mcp_servers=mcp_servers,
                on_delta=usage_tracker.on_delta,
                local=LocalSendOptions(force=True) if force else None,
            ),
        )
        result = await collect_run_result(
            run, max_turns=max_turns, usage_tracker=usage_tracker, on_tool=on_tool,
        )
        result.session_id = self.agent.agent_id
        return result

    async def close(self) -> None:
        """Release the agent's executor. Its conversation stays on disk."""
        agent, self.agent, self.client = self.agent, None, None
        if agent is None:
            return
        try:
            await agent.close()
        except Exception as exc:
            logger.warning("Error while closing Cursor agent: {exc}", exc=exc)
