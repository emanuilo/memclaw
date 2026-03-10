from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from loguru import logger

from .config import MemclawConfig
from .index import MemoryIndex
from .search import HybridSearch, SearchResult
from .store import MemoryStore

SYSTEM_PROMPT = """\
Today's date: {today}

You are Memclaw, a personal memory assistant. You help users store and retrieve \
their thoughts, notes, ideas, and images.

Capabilities:
1. **Store**: When the user shares information, thoughts, notes, facts, or anything \
worth remembering — save it using the memory_save tool. Briefly confirm what you saved.
2. **Search**: When the user asks a question or wants to recall something — search \
their memories using memory_search. Present results clearly with dates.
3. **Images (local file)**: When the user provides a local image file path, describe \
and save it with the image_save tool.
4. **Images (Telegram)**: When you see an image in the message with a file_id, \
describe what you see in detail and save the description using telegram_image_save \
with the provided file_id.
5. **Image retrieval**: When the user asks to retrieve, show, or find an image — use \
image_search. The image will be sent automatically, just acknowledge it briefly.
6. **Conversation**: Sometimes the user just wants to chat. Respond naturally. If they \
mention something worth remembering, save it too.

You may also receive pre-processed content:
- "[Voice message]" followed by a transcription — the voice is already transcribed \
and stored. Respond to the content.
- "[Link summary]" entries — links have been fetched and summarized for you.

=== MEMORY CONTEXT ===
{context}

=== GUIDELINES ===
- Always respond to the user. Never be silent.
- Be concise and helpful.
- When storing, briefly confirm what was saved.
- When searching, present the most relevant results clearly with source dates.
- If intent is ambiguous, lean towards storing when sharing info and searching \
when asking questions.
- Reference specific memories with dates when relevant.
- If information conflicts, prefer more recent data.
"""


def _format_results(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        if source == "MEMORY":
            source = "permanent memory"
        parts.append(f"**[{i}]** (score: {r.score:.2f}, source: {source})\n{r.content.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "No matching memories found."


class MemclawAgent:
    """Unified agent for both interactive CLI and Telegram bot.

    Every message goes through ``handle()``, which builds memory context,
    runs the Claude Agent SDK loop, and returns the response text together
    with any found images (relevant for Telegram image retrieval).
    """

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)
        self.search = HybridSearch(config, self.index)
        self._found_images: list[dict] = []

        tools = self._create_tools()
        self.server = create_sdk_mcp_server(name="memclaw", version="0.1.0", tools=tools)
        self.tool_names = [
            "mcp__memclaw__memory_save",
            "mcp__memclaw__memory_search",
            "mcp__memclaw__image_save",
            "mcp__memclaw__telegram_image_save",
            "mcp__memclaw__image_search",
        ]

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _create_tools(self):
        store = self.store
        index = self.index
        search = self.search
        found_images = self._found_images

        @tool("memory_save", "Save a new memory, thought, or note", {"content": str})
        async def memory_save_tool(args):
            content: str = args["content"]
            permanent: bool = args.get("permanent", False)
            entry_type: str = args.get("entry_type", "note")
            tags = args.get("tags")

            file_path = store.save(content, permanent=permanent, entry_type=entry_type, tags=tags)
            await index.index_file(file_path)
            logger.info("  → memory_save result: saved to {file}", file=file_path.name)
            return {"content": [{"type": "text", "text": f"Memory saved to {file_path.name}"}]}

        @tool(
            "memory_search",
            "Search through stored memories using natural language",
            {"query": str},
        )
        async def memory_search_tool(args):
            results = await search.search(args["query"], limit=args.get("limit", 10))
            formatted = _format_results(results)
            logger.info("  → memory_search result: {n} hits", n=len(results))
            for i, r in enumerate(results, 1):
                source = Path(r.file_path).stem
                snippet = r.content.strip().replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                logger.info("    [{i}] ({score:.2f}, {src}) {snippet}", i=i, score=r.score, src=source, snippet=snippet)
            return {"content": [{"type": "text", "text": formatted}]}

        @tool(
            "image_save",
            "Save a local image by generating an AI description and storing it as a memory. "
            "You can see the image — describe it yourself and pass your description as content.",
            {"image_path": str},
        )
        async def image_save_tool(args):
            image_path = Path(args["image_path"]).expanduser().resolve()
            caption: str = args.get("caption", "")

            if not image_path.exists():
                logger.info("  → image_save result: not found {path}", path=image_path)
                return {"content": [{"type": "text", "text": f"Image not found: {image_path}"}]}

            memory_content = f"**Image:** {image_path.name}\n"
            if caption:
                memory_content += f"**Caption:** {caption}\n"
            memory_content += f"**Path:** {image_path}\n"

            file_path = store.save(memory_content, entry_type="image")
            await index.index_file(file_path)

            logger.info("  → image_save result: saved {name}", name=image_path.name)
            return {"content": [{"type": "text", "text": f"Image saved from {image_path.name}"}]}

        @tool(
            "telegram_image_save",
            "Save a Telegram image with your description for later retrieval. "
            "You MUST call this when you receive an image with a file_id. Describe the image in "
            "detail and pass the description along with the file_id from the message.",
            {"description": str, "file_id": str},
        )
        async def telegram_image_save_tool(args):
            description: str = args["description"]
            file_id: str = args["file_id"]
            caption: str = args.get("caption", "")

            combined = f"Image: {description}"
            if caption:
                combined += f" Caption: {caption}"

            file_path = store.save(combined, entry_type="image")
            await index.index_file(file_path)

            await index.store_telegram_image(
                file_id=file_id,
                description=combined,
                caption=caption,
            )

            logger.info("  → telegram_image_save result: {desc}", desc=description[:100])
            return {"content": [{"type": "text", "text": f"Image saved: {description[:100]}"}]}

        @tool(
            "image_search",
            "Search for previously stored images to send to the user. "
            "Use when the user asks to retrieve, show, or send an image. "
            "The image will be sent automatically.",
            {"query": str},
        )
        async def image_search_tool(args):
            query_emb = await index.get_embedding(args["query"])
            results = index.search_telegram_images(query_emb, limit=3)
            found_images.extend(results)

            if results:
                lines = []
                for r in results:
                    line = f"- {r['description']}"
                    if r.get("caption"):
                        line += f" (caption: {r['caption']})"
                    lines.append(line)
                logger.info("  → image_search result: {n} image(s) found", n=len(results))
                for line in lines:
                    logger.info("    {line}", line=line)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Found {len(results)} image(s):\n"
                                + "\n".join(lines)
                                + "\nImages will be sent automatically."
                            ),
                        }
                    ]
                }
            logger.info("  → image_search result: no matching images")
            return {"content": [{"type": "text", "text": "No matching images found."}]}

        return [
            memory_save_tool,
            memory_search_tool,
            image_save_tool,
            telegram_image_save_tool,
            image_search_tool,
        ]

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    async def build_context(self, message: str) -> str:
        """Build memory context to inject into the system prompt."""
        parts: list[str] = []

        # Always include permanent memory
        memory_content = self.store.read_file(self.config.memory_file)
        if memory_content.strip():
            parts.append("### Permanent Memory")
            parts.append(memory_content[:3000])

        # Semantic search for message-relevant memories
        results = await self.search.search(message, limit=10)
        if results:
            parts.append("\n### Relevant Memories")
            for r in results:
                source = Path(r.file_path).stem
                parts.append(f"[{source}] {r.content.strip()}")

        return "\n\n".join(parts) if parts else "No memories found yet."

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
    ) -> tuple[str, list[dict]]:
        """Process any message through the agent.

        Works identically for console and Telegram modes.

        Args:
            message: Text message or prompt to send.
            image_b64: Optional base64-encoded image data.
            image_media_type: MIME type of the image (default: image/jpeg).

        Returns:
            (response_text, found_images) — found_images is a list of dicts
            with Telegram file_ids; empty in console mode.
        """
        self._found_images.clear()

        context = await self.build_context(message)

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT.format(
                today=date.today().isoformat(),
                context=context,
            ),
            mcp_servers={"memclaw": self.server},
            allowed_tools=self.tool_names,
            permission_mode="bypassPermissions",
            max_turns=10,
        )

        # Build content blocks — text only, or text + image
        if image_b64:
            content_blocks: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": message},
            ]
        else:
            content_blocks = None

        last_text = ""
        async with ClaudeSDKClient(options=options) as client:
            if content_blocks is not None:
                async def _image_msg() -> AsyncIterator[dict[str, Any]]:
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": content_blocks},
                    }

                await client.query(_image_msg())
            else:
                await client.query(message)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "name") and hasattr(block, "input"):
                            # ToolUseBlock
                            args_str = json.dumps(block.input, ensure_ascii=False)
                            if len(args_str) > 300:
                                args_str = args_str[:300] + "..."
                            logger.info("Tool call: {name}({args})", name=block.name, args=args_str)
                        elif hasattr(block, "text"):
                            last_text = block.text
                elif isinstance(msg, ResultMessage):
                    if hasattr(msg, "result") and msg.result:
                        last_text = msg.result
                    cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "n/a"
                    logger.info(
                        "Agent done: {turns} turns, {ms}ms, cost {cost}",
                        turns=msg.num_turns,
                        ms=msg.duration_ms,
                        cost=cost,
                    )

        return (
            last_text or "I couldn't generate a response.",
            list(self._found_images),
        )

    def close(self):
        self.index.close()
