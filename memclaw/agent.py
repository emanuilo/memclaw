"""Memclaw agent — backed by claude-agent-sdk (claude CLI subprocess).

Auth precedence is managed via ClaudeAgentOptions.env: we strip ANTHROPIC_API_KEY
and ANTHROPIC_AUTH_TOKEN and set CLAUDE_CODE_OAUTH_TOKEN so requests bill against
the user's Claude subscription (Max plan + Extra usage) rather than the API
console credit balance.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from loguru import logger

from .config import MemclawConfig
from .index import MemoryIndex
from .reminders import ReminderScheduler
from .search import HybridSearch
from .store import MemoryStore
from .tools import MCP_SERVER_NAME, TOOL_DEFINITIONS, ToolExecutor, build_mcp_server

# ── Prompts ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
Today's date: {today}
Current local time: {now}

{agent_instructions}

=== REPLY FORMATTING ===
Replies are delivered to messaging apps with limited markdown support. Use \
ONLY this minimal syntax — anything else leaks as literal characters:
- Bold: `*bold*` (single asterisk). NEVER use `**double asterisks**`.
- Italic: `_italic_`.
- Bullet lists: plain `- item` on its own line.
- Paragraphs: separate with a blank line.
- Headings (`#`, `##`, ...) are NOT supported — use a bold line on its own \
(e.g. `*Section name*`) followed by a blank line instead.
- No backticks or fenced code blocks.
- No `[label](url)` links — write the bare URL.

=== MEMORY CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{history}

IMPORTANT: When the user gives you a behavioural instruction (e.g. "always respond \
in Spanish", "be more formal", "never use emojis"), you MUST call the \
update_instructions tool to save it. These are rules you should follow in every \
future conversation.
"""


def _load_agent_instructions(config: MemclawConfig) -> str:
    agent_file = config.agent_file
    if agent_file.exists():
        return agent_file.read_text().strip()
    return "You are Memclaw, a personal memory assistant."


_CONSOLIDATION_PROMPT = """\
You are a memory consolidation assistant. Your job is to distill daily memory \
logs into a curated, permanent knowledge base.

You will receive:
1. The content of several daily memory files (chronological notes, thoughts, \
saved links, voice transcriptions, etc.)
2. The current content of MEMORY.md (the permanent memory file), which may be \
empty if this is the first consolidation.

Your task:
- Extract durable facts, preferences, decisions, and important events from the \
daily files.
- Ignore transient entries: one-off reminders that have passed, trivial \
greetings, temporary notes, etc.
- Merge the extracted information with the existing MEMORY.md content. Update \
existing entries if new information supersedes them. Remove outdated entries.
- Output the complete updated MEMORY.md content in structured markdown with \
sections such as:
  ## Preferences
  ## Projects
  ## People
  ## Key Facts
  ## Decisions
  ## Important Events
- Only include sections that have content. You may add other sections if \
appropriate.
- Place the most important and frequently referenced information at the top.
- Keep the output concise — target under 5,000 characters.
- Output ONLY the markdown content for MEMORY.md. Do not include any \
explanation or preamble.
"""

# Sonnet 4 pricing (per 1M tokens) — used as fallback when the SDK doesn't
# return a total_cost_usd, and for the "(in=..., out=..., cache_read=...)"
# log line so it mirrors the old raw-API log format.
_INPUT_COST_PER_M = 3.0
_OUTPUT_COST_PER_M = 15.0

_MODEL = "claude-sonnet-4-6"

# All Claude Code built-in tools — disabled so the agent only uses our MCP tools.
_BUILTIN_TOOLS_DISALLOW = [
    "Bash", "BashOutput", "KillBash",
    "Read", "Write", "Edit", "NotebookEdit",
    "Grep", "Glob",
    "Task",
    "WebFetch", "WebSearch",
    "TodoWrite",
    "SlashCommand", "ExitPlanMode",
]

# Pre-approved MCP tool names as Claude sees them.
_ALLOWED_TOOLS = [
    f"mcp__{MCP_SERVER_NAME}__{t['name']}" for t in TOOL_DEFINITIONS
]


def _build_env(oauth_token: str) -> dict[str, str]:
    """Build the env dict for the Claude CLI subprocess.

    Scrub ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN so the OAuth token wins
    precedence and requests bill against the Max subscription, not API credits.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
    }
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


class MemclawAgent:
    """Unified agent for both interactive CLI and messaging bots.

    Delegates every turn to the `claude` CLI via claude-agent-sdk, which:
    - Authenticates against the user's Claude subscription (Max / Extra usage).
    - Runs an in-process MCP server exposing the memclaw tool suite.
    - Handles prompt caching, tool loops, and session persistence itself.
    """

    def __init__(
        self,
        config: MemclawConfig,
        platform: str | None = None,
        *,
        scheduler: ReminderScheduler | None = None,
    ):
        self.config = config
        self.platform = platform
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self.scheduler = scheduler
        self._found_images: list[dict] = []
        self._history: list[dict] = []
        self._tools = ToolExecutor(
            config=config,
            store=self.store,
            index=self.index,
            search=self.search,
            found_images=self._found_images,
            platform=platform,
            scheduler=scheduler,
        )
        self._mcp_server = build_mcp_server(self._tools)
        self._env = _build_env(config.claude_code_oauth_token)
        # Keep a stable session id so the CLI's prompt cache stays warm across
        # turns for this agent instance.
        self._session_id = f"memclaw-{platform or 'cli'}"

    # ── Startup / sync ───────────────────────────────────────────────

    async def start(self):
        await self.index.sync()

    async def start_background_sync(self, interval: int = 60):
        index = self.index

        async def _sync_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await index.sync()
                except Exception:
                    pass

        self._sync_task = asyncio.create_task(_sync_loop())

    # ── Consolidation ────────────────────────────────────────────────

    async def _maybe_consolidate(
        self,
        *,
        force: bool = False,
        consolidated_through_override: date | None = None,
    ) -> bool:
        meta_path = self.config.memory_dir / "meta.json"

        consolidated_through: date | None = None
        if consolidated_through_override is not None:
            consolidated_through = consolidated_through_override
        elif meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                ct = meta.get("consolidated_through")
                if ct:
                    consolidated_through = date.fromisoformat(ct)
            except (json.JSONDecodeError, ValueError):
                pass

        unconsolidated = self.store.list_unconsolidated_files(consolidated_through)
        if not unconsolidated:
            return False
        if len(unconsolidated) < self.config.consolidation_threshold and not force:
            return False

        daily_content_parts: list[str] = []
        total_chars = 0
        for path in unconsolidated:
            content = self.store.read_file(path)
            if not content.strip():
                continue
            header = f"\n### {path.stem}\n\n"
            chunk = header + content
            if total_chars + len(chunk) > 30000:
                remaining = 30000 - total_chars
                if remaining > 0:
                    daily_content_parts.append(chunk[:remaining])
                break
            daily_content_parts.append(chunk)
            total_chars += len(chunk)

        daily_text = "\n".join(daily_content_parts)
        if not daily_text.strip():
            return False

        existing_memory = self.store.read_file(self.config.memory_file)
        user_message = "## Daily Memory Files\n\n" + daily_text
        if existing_memory.strip():
            user_message += "\n\n## Current MEMORY.md\n\n" + existing_memory

        options = ClaudeAgentOptions(
            env=self._env,
            model=_MODEL,
            system_prompt=_CONSOLIDATION_PROMPT,
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

        if not result_text.strip():
            return False

        self.config.memory_file.write_text(result_text)

        last_date_str = unconsolidated[-1].stem
        try:
            new_consolidated_through = date.fromisoformat(last_date_str)
        except ValueError:
            new_consolidated_through = date.today()

        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, ValueError):
                pass
        meta["consolidated_through"] = new_consolidated_through.isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))

        await self.index.index_file(self.config.memory_file)
        logger.info(
            "Consolidation complete: {n} files → MEMORY.md (through {d})",
            n=len(unconsolidated),
            d=new_consolidated_through.isoformat(),
        )
        return True

    # ── Context builder ──────────────────────────────────────────────

    async def build_context(self, message: str) -> str:
        parts: list[str] = []

        memory_content = self.store.read_file(self.config.memory_file)
        if memory_content.strip():
            parts.append("### Permanent Memory")
            if len(memory_content) <= 4000:
                parts.append(memory_content)
            else:
                parts.append(memory_content[:2000])
                memory_results = await self.search.search(
                    message, limit=3, file_filter="MEMORY.md"
                )
                if memory_results:
                    parts.append("\n#### Relevant Permanent Memory Sections")
                    for r in memory_results:
                        parts.append(r.content.strip())

        results = await self.search.search(message, limit=10)
        if results:
            parts.append("\n### Relevant Memories")
            for r in results:
                source = Path(r.file_path).stem
                parts.append(f"[{source}] {r.content.strip()}")

        return "\n\n".join(parts) if parts else "No memories found yet."

    # ── Main entry point ─────────────────────────────────────────────

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        chat_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        self._found_images.clear()
        self._tools.chat_id = chat_id

        try:
            await self._maybe_consolidate()
        except Exception as exc:
            logger.warning("Consolidation check failed: {exc}", exc=exc)

        history_content = "[User sent a photo]" if image_b64 else message
        self._history.append({
            "role": "user",
            "content": history_content,
            "timestamp": datetime.now().isoformat(),
        })

        context = await self.build_context(message)

        history_snapshot = self._history[:-1]
        if history_snapshot:
            history_lines = []
            for entry in history_snapshot:
                role = "User" if entry["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {entry['content']}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "(no prior messages)"

        agent_instructions = _load_agent_instructions(self.config)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            today=date.today().isoformat(),
            now=datetime.now().replace(microsecond=0).isoformat(),
            agent_instructions=agent_instructions,
            context=context,
            history=history_text,
        )

        options = ClaudeAgentOptions(
            env=self._env,
            model=_MODEL,
            system_prompt=system_prompt,
            setting_sources=None,
            mcp_servers={MCP_SERVER_NAME: self._mcp_server},
            allowed_tools=_ALLOWED_TOOLS,
            disallowed_tools=_BUILTIN_TOOLS_DISALLOW,
            permission_mode="bypassPermissions",
            max_turns=10,
            # Every handle() call is a fresh conversation — the system prompt
            # carries the relevant history snapshot. Giving the client a fresh
            # session id each turn also isolates the CLI's own transcript from
            # our in-process history accounting.
        )

        t0 = time.perf_counter()
        last_text = ""
        num_turns = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_creation_tokens = 0
        total_cost_usd: float | None = None

        async with ClaudeSDKClient(options=options) as client:
            if image_b64:
                await client.query(_image_prompt_stream(
                    message, image_b64, image_media_type,
                ))
            else:
                await client.query(message)

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
                            if tool_name.startswith(f"mcp__{MCP_SERVER_NAME}__"):
                                tool_name = tool_name[len(f"mcp__{MCP_SERVER_NAME}__"):]
                            logger.info("Tool call: {name}({args})", name=tool_name, args=args_str)
                    if turn_text:
                        last_text = turn_text
                elif isinstance(msg, ResultMessage):
                    num_turns = msg.num_turns
                    total_cost_usd = msg.total_cost_usd
                    if msg.usage:
                        total_input_tokens = msg.usage.get("input_tokens", 0) or 0
                        total_output_tokens = msg.usage.get("output_tokens", 0) or 0
                        total_cache_read_tokens = msg.usage.get("cache_read_input_tokens", 0) or 0
                        total_cache_creation_tokens = msg.usage.get("cache_creation_input_tokens", 0) or 0
                    if msg.result and not last_text:
                        last_text = msg.result

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Prefer the SDK's client-side cost estimate when available; else
        # compute from tokens the way the raw-API loop used to.
        if total_cost_usd is None:
            cache_read_cost = total_cache_read_tokens * _INPUT_COST_PER_M * 0.1 / 1_000_000
            cost = (
                total_input_tokens * _INPUT_COST_PER_M / 1_000_000
                + total_output_tokens * _OUTPUT_COST_PER_M / 1_000_000
                + cache_read_cost
            )
        else:
            cost = total_cost_usd

        logger.info(
            "Agent done: {turns} turns, {ms}ms, cost ${cost:.4f} "
            "(in={input_t}, out={output_t}, cache_read={cache_r}, cache_create={cache_c})",
            turns=num_turns or 1,
            ms=elapsed_ms,
            cost=cost,
            input_t=total_input_tokens,
            output_t=total_output_tokens,
            cache_r=total_cache_read_tokens,
            cache_c=total_cache_creation_tokens,
        )

        response_text = last_text or "I couldn't generate a response."
        self._history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
        })

        max_entries = self.config.conversation_history_limit * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        return (response_text, list(self._found_images))

    def close(self):
        task = getattr(self, "_sync_task", None)
        if task is not None and not task.done():
            task.cancel()
        self.index.close()


async def _image_prompt_stream(
    message: str, image_b64: str, image_media_type: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield a single streaming-input user message containing an image + text.

    The Claude CLI's stream-json protocol expects Anthropic-style content
    blocks here, so we pass an "image" block with a base64 source followed
    by the user's text.
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
