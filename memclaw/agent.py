from __future__ import annotations

import base64
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

from .config import MemclawConfig
from .index import MemoryIndex
from .search import HybridSearch, SearchResult
from .store import MemoryStore

SYSTEM_PROMPT = """\
You are Memclaw, a personal memory assistant. You help users store and retrieve \
their thoughts, notes, ideas, and images.

Capabilities:
- **Store**: When a user shares a thought, fact, note, or information, save it \
with the memory_save tool.
- **Search**: When a user asks a question or wants to recall something, search \
their memories with the memory_search tool.
- **Images**: When a user provides an image file path, describe and save it with \
the image_save tool.

Guidelines:
- Be concise and helpful.
- When storing, briefly confirm what was saved.
- When searching, present the most relevant results clearly with source dates.
- If intent is ambiguous, lean towards storing when sharing info and searching \
when asking questions.

Today's date: {today}
"""


def _format_results(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        source = Path(r.file_path).stem
        if source == "MEMORY":
            source = "permanent memory"
        parts.append(f"**[{i}]** (score: {r.score:.2f}, source: {source})\n{r.content.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "No matching memories found."


def create_memclaw_tools(config: MemclawConfig):
    """Build MCP tools backed by the core memory engine.

    Returns (tools_list, components_dict) so callers can access internals
    (e.g. to close the index).
    """
    store = MemoryStore(config)
    index = MemoryIndex(config)
    search = HybridSearch(config, index)

    @tool("memory_save", "Save a new memory, thought, or note", {"content": str})
    async def memory_save(args):
        content: str = args["content"]
        permanent: bool = args.get("permanent", False)
        entry_type: str = args.get("entry_type", "note")
        tags = args.get("tags")

        file_path = store.save(content, permanent=permanent, entry_type=entry_type, tags=tags)
        await index.index_file(file_path)
        return {"content": [{"type": "text", "text": f"Memory saved to {file_path.name}"}]}

    @tool(
        "memory_search",
        "Search through stored memories using natural language",
        {"query": str},
    )
    async def memory_search_tool(args):
        query_text: str = args["query"]
        limit: int = args.get("limit", 10)

        results = await search.search(query_text, limit=limit)
        formatted = _format_results(results)
        return {"content": [{"type": "text", "text": formatted}]}

    @tool(
        "image_save",
        "Save an image by generating an AI description and storing it as a memory. "
        "You can see the image — describe it yourself and pass your description as content.",
        {"image_path": str},
    )
    async def image_save(args):
        image_path = Path(args["image_path"]).expanduser().resolve()
        caption: str = args.get("caption", "")

        if not image_path.exists():
            return {"content": [{"type": "text", "text": f"Image not found: {image_path}"}]}

        image_data = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = {
            "jpg": "jpeg",
            "jpeg": "jpeg",
            "png": "png",
            "gif": "gif",
            "webp": "webp",
        }.get(suffix, "jpeg")

        memory_content = f"**Image:** {image_path.name}\n"
        if caption:
            memory_content += f"**Caption:** {caption}\n"
        memory_content += f"**Path:** {image_path}\n"

        file_path = store.save(memory_content, entry_type="image")
        await index.index_file(file_path)

        return {"content": [{"type": "text", "text": f"Image saved from {image_path.name}"}]}

    tools = [memory_save, memory_search_tool, image_save]
    components = {"store": store, "index": index, "search": search}
    return tools, components


class MemclawAgent:
    """High-level agent that wraps the Claude Agent SDK using ClaudeSDKClient."""

    TOOL_NAMES = [
        "mcp__memclaw__memory_save",
        "mcp__memclaw__memory_search",
        "mcp__memclaw__image_save",
    ]

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.tools, self.components = create_memclaw_tools(config)
        self.server = create_sdk_mcp_server(name="memclaw", version="0.1.0", tools=self.tools)

    async def chat(self, prompt: str) -> str:
        """Send a message to the agent and return its text response."""
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
            mcp_servers={"memclaw": self.server},
            allowed_tools=self.TOOL_NAMES,
            permission_mode="bypassPermissions",
            max_turns=10,
        )

        last_text = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            last_text = block.text
                elif isinstance(message, ResultMessage) and hasattr(message, "result"):
                    last_text = message.result or last_text

        return last_text

    def close(self):
        self.components["index"].close()


# ======================================================================
# Telegram-specific agent
# ======================================================================

TELEGRAM_SYSTEM_PROMPT = """\
Today's date: {today}

You are Memclaw, a personal memory assistant on Telegram. Every message the user \
sends comes to you. You must decide what to do based on the user's intent:

1. **Store**: When the user shares information, thoughts, notes, facts, or anything \
worth remembering — save it using the memory_save tool. Briefly confirm what you saved.
2. **Search**: When the user asks a question or wants to recall something — search \
their memories using memory_search. Present results clearly with dates.
3. **Images**: When the user asks to retrieve, show, or find an image — use \
image_search. The image will be sent automatically, just acknowledge it briefly.
4. **Image received**: When you see an image in the message, describe what you see \
in detail and save the description using telegram_image_save with the provided file_id. \
Always confirm what you saved.
5. **Conversation**: Sometimes the user just wants to chat. Respond naturally. If they \
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
- When intent is ambiguous, lean towards storing when sharing info and searching \
when asking questions.
- Reference specific memories with dates when relevant.
- If information conflicts, prefer more recent data.
"""


class TelegramAgent:
    """Agent for the Telegram bot — every message goes through this agent."""

    def __init__(
        self,
        config: MemclawConfig,
        store: MemoryStore,
        index: MemoryIndex,
        search_engine: HybridSearch,
    ):
        self.config = config
        self.store = store
        self.index = index
        self.search = search_engine
        self._found_images: list[dict] = []

        tools = self._create_tools()
        self.server = create_sdk_mcp_server(name="memclaw-tg", version="0.1.0", tools=tools)
        self.tool_names = [
            "mcp__memclaw-tg__memory_save",
            "mcp__memclaw-tg__memory_search",
            "mcp__memclaw-tg__image_search",
            "mcp__memclaw-tg__telegram_image_save",
        ]

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
            return {"content": [{"type": "text", "text": f"Memory saved to {file_path.name}"}]}

        @tool(
            "telegram_image_save",
            "Save a Telegram image with your description for later retrieval. "
            "You MUST call this when you receive an image. Describe the image in "
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

            # Save to markdown memory
            file_path = store.save(combined, entry_type="image")
            await index.index_file(file_path)

            # Save to telegram image registry for file_id retrieval
            await index.store_telegram_image(
                file_id=file_id,
                description=combined,
                caption=caption,
            )

            return {"content": [{"type": "text", "text": f"Image saved: {description[:100]}"}]}

        @tool(
            "memory_search",
            "Search through stored memories using natural language",
            {"query": str},
        )
        async def memory_search_tool(args):
            results = await search.search(args["query"], limit=args.get("limit", 10))
            formatted = _format_results(results)
            return {"content": [{"type": "text", "text": formatted}]}

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
            return {"content": [{"type": "text", "text": "No matching images found."}]}

        return [memory_save_tool, telegram_image_save_tool, memory_search_tool, image_search_tool]

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

    async def handle(
        self,
        message: str,
        *,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
    ) -> tuple[str, list[dict]]:
        """Process any message through the agent.

        Args:
            message: Text message or prompt to send.
            image_b64: Optional base64-encoded image data. When provided,
                the agent sees the image directly and can describe it.
            image_media_type: MIME type of the image (default: image/jpeg).

        Returns (response_text, found_images).
        """
        self._found_images.clear()

        context = await self.build_context(message)

        options = ClaudeAgentOptions(
            system_prompt=TELEGRAM_SYSTEM_PROMPT.format(
                today=date.today().isoformat(),
                context=context,
            ),
            mcp_servers={"memclaw-tg": self.server},
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
            content_blocks = message  # type: ignore[assignment]

        last_text = ""
        async with ClaudeSDKClient(options=options) as client:
            if isinstance(content_blocks, list):
                # Multimodal: send as async iterable with content blocks
                async def _image_msg() -> AsyncIterator[dict[str, Any]]:
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": content_blocks},
                    }

                await client.query(_image_msg())
            else:
                # Text only
                await client.query(content_blocks)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            last_text = block.text
                elif isinstance(msg, ResultMessage) and hasattr(msg, "result"):
                    last_text = msg.result or last_text

        return (
            last_text or "I couldn't generate a response.",
            list(self._found_images),
        )
