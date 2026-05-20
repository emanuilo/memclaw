"""Claude Agent SDK backend for Memclaw.

Wraps `claude-agent-sdk` (which itself shells out to the Claude CLI) into the
neutral `AgentBackend` protocol. All Claude-specific glue — env scrubbing,
MCP server wrapping for tools, stream-json image protocol, response parsing
— lives in this file.
"""

from __future__ import annotations

import json
import os
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
from rich.panel import Panel
from rich.prompt import Prompt

from ..tools import TOOL_DEFINITIONS
from .base import TurnResult

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor


_MODEL = "claude-sonnet-4-6"

# Sonnet 4 pricing (per 1M tokens) — only used as a fallback if the SDK
# doesn't return total_cost_usd for an API-key turn.
_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0

# Claude Code built-ins are disabled so the agent only sees our MCP tools.
_BUILTIN_TOOLS_DISALLOW = [
    "Bash", "BashOutput", "KillBash",
    "Read", "Write", "Edit", "NotebookEdit",
    "Grep", "Glob",
    "Task",
    "WebFetch", "WebSearch",
    "TodoWrite",
    "SlashCommand", "ExitPlanMode",
]

MCP_SERVER_NAME = "memclaw"
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

async def _image_prompt_stream(
    message: str, image_b64: str, image_media_type: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield a single streaming-input user message containing an image + text.

    The Claude CLI's stream-json protocol expects Anthropic-style content
    blocks here, so we pass an "image" block with a base64 source followed
    by the user's text. Targets claude-agent-sdk 0.1.x.
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

class ClaudeAgentBackend:
    """Claude Agent SDK implementation of the AgentBackend protocol."""

    name: ClassVar[str] = "claude"
    display_name: ClassVar[str] = "Claude Agent SDK"

    def __init__(self, config: "MemclawConfig") -> None:
        self.config = config
        self._env = _build_env(config)
        self._mcp_server: Any = None  # built lazily, needs a ToolExecutor
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
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        """Ask whether to use a subscription or API key, then prompt the
        chosen credential. Returns (values_to_save, env_keys_to_drop).
        """
        console.print()
        console.print(
            Panel(
                "[bold]1)[/bold] Claude subscription (Pro / Max / Team)\n"
                "    No per-message cost — uses your subscription quota.\n"
                "    Generate a token with: [bold]claude setup-token[/bold]\n\n"
                "[bold]2)[/bold] Anthropic API key (pay-as-you-go)\n"
                "    Billed per token against your console credits.\n"
                "    Get a key at: console.anthropic.com",
                title="How do you want to authenticate with Claude?",
                border_style="bright_cyan",
            )
        )

        if existing.get("ANTHROPIC_API_KEY") and not existing.get("CLAUDE_CODE_OAUTH_TOKEN"):
            default = "2"
        else:
            default = "1"
        choice = Prompt.ask("Choose", choices=["1", "2"], default=default)

        if choice == "1":
            env_key = "CLAUDE_CODE_OAUTH_TOKEN"
            label = "Claude subscription OAuth token (run `claude setup-token`)"
            drop_key = "ANTHROPIC_API_KEY"
        else:
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

        return {env_key: value}, [drop_key]

    # -- Runtime ---------------------------------------------------------

    async def run_one_shot(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        options = ClaudeAgentOptions(
            env=self._env,
            model=_MODEL,
            system_prompt=system_prompt,
            setting_sources=None,
            disallowed_tools=_BUILTIN_TOOLS_DISALLOW,
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
        user_message: str,
        tool_executor: "ToolExecutor",
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
    ) -> TurnResult:
        if self._mcp_server is None:
            self._mcp_server = _build_mcp_server(tool_executor)

        options = ClaudeAgentOptions(
            env=self._env,
            model=_MODEL,
            system_prompt=system_prompt,
            setting_sources=None,
            mcp_servers={MCP_SERVER_NAME: self._mcp_server},
            allowed_tools=_ALLOWED_TOOLS,
            disallowed_tools=_BUILTIN_TOOLS_DISALLOW,
            # SAFETY: bypassPermissions is only safe because allowed_tools
            # restricts execution to mcp__memclaw__* (our in-process server)
            # and _BUILTIN_TOOLS_DISALLOW blocks Claude Code's built-ins.
            # If either guardrail is loosened, revisit this — bypass mode
            # would otherwise turn any future broad tool into an RCE vector.
            permission_mode="bypassPermissions",
            max_turns=max_turns,
        )

        last_text = ""
        result = TurnResult(text="")

        async with ClaudeSDKClient(options=options) as client:
            if image_b64:
                await client.query(_image_prompt_stream(
                    user_message, image_b64, image_media_type,
                ))
            else:
                await client.query(user_message)

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
                    result.num_turns = msg.num_turns
                    result.cost_usd = msg.total_cost_usd
                    if msg.usage:
                        result.input_tokens = msg.usage.get("input_tokens", 0) or 0
                        result.output_tokens = msg.usage.get("output_tokens", 0) or 0
                        result.cache_read_tokens = msg.usage.get("cache_read_input_tokens", 0) or 0
                        result.cache_creation_tokens = msg.usage.get("cache_creation_input_tokens", 0) or 0
                    if msg.result and not last_text:
                        last_text = msg.result

        # Fallback cost estimate when the SDK didn't supply one.
        if result.cost_usd is None and self.bills_per_token:
            cache_read_cost = result.cache_read_tokens * _INPUT_COST_PER_M * 0.1 / 1_000_000
            result.cost_usd = (
                result.input_tokens * _INPUT_COST_PER_M / 1_000_000
                + result.output_tokens * _OUTPUT_COST_PER_M / 1_000_000
                + cache_read_cost
            )

        result.text = last_text
        return result
