from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from openai import AsyncOpenAI

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
        parts.append(
            f"**[{i}]** (score: {r.score:.2f}, source: {source})\n{r.content.strip()}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No matching memories found."


def create_memclaw_tools(config: MemclawConfig):
    """Build MCP tools backed by the core memory engine.

    Returns (tools_list, components_dict) so callers can access internals
    (e.g. to close the index).
    """
    store = MemoryStore(config)
    index = MemoryIndex(config)
    search = HybridSearch(config, index)
    openai_client = AsyncOpenAI(api_key=config.openai_api_key)

    @tool("memory_save", "Save a new memory, thought, or note", {"content": str})
    async def memory_save(args):
        content: str = args["content"]
        permanent: bool = args.get("permanent", False)
        entry_type: str = args.get("entry_type", "note")
        tags = args.get("tags")

        file_path = store.save(
            content, permanent=permanent, entry_type=entry_type, tags=tags
        )
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
        "Save an image by generating an AI description and storing it as a memory",
        {"image_path": str},
    )
    async def image_save(args):
        image_path = Path(args["image_path"]).expanduser().resolve()
        caption: str = args.get("caption", "")

        if not image_path.exists():
            return {
                "content": [{"type": "text", "text": f"Image not found: {image_path}"}]
            }

        image_data = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = {
            "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "gif": "gif", "webp": "webp",
        }.get(suffix, "jpeg")

        prompt_text = "Describe this image in detail in about 50 words."
        if caption:
            prompt_text += f" Context: {caption}"

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{media_type};base64,{image_data}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            max_tokens=150,
        )
        description = response.choices[0].message.content

        memory_content = f"**Image:** {image_path.name}\n"
        if caption:
            memory_content += f"**Caption:** {caption}\n"
        memory_content += f"**Description:** {description}\n"
        memory_content += f"**Path:** {image_path}\n"

        file_path = store.save(memory_content, entry_type="image")
        await index.index_file(file_path)

        return {"content": [{"type": "text", "text": f"Image saved: {description}"}]}

    tools = [memory_save, memory_search_tool, image_save]
    components = {"store": store, "index": index, "search": search}
    return tools, components


class MemclawAgent:
    """High-level agent that wraps the Claude Agent SDK."""

    TOOL_NAMES = [
        "mcp__memclaw__memory_save",
        "mcp__memclaw__memory_search",
        "mcp__memclaw__image_save",
    ]

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.tools, self.components = create_memclaw_tools(config)
        self.server = create_sdk_mcp_server(
            name="memclaw", version="0.1.0", tools=self.tools
        )
        self._session_id: str | None = None

    async def chat(self, prompt: str) -> str:
        """Send a message to the agent and return its text response."""
        opts: dict = dict(
            system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
            mcp_servers={"memclaw": self.server},
            allowed_tools=self.TOOL_NAMES,
            permission_mode="bypassPermissions",
            max_turns=10,
        )

        if self._session_id:
            opts["resume"] = self._session_id

        options = ClaudeAgentOptions(**opts)

        last_text = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "subtype") and message.subtype == "init":
                self._session_id = message.session_id
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

You are a personal AI assistant with access to the user's memories.

=== MEMORY CONTEXT ===
{context}

=== INSTRUCTIONS ===
- Use the memory context above to answer questions.
- If you need more specific memories, use the memory_search tool.
- When the user asks to retrieve, show, or send an image, use the image_search tool.
  The image will be sent automatically — just acknowledge it briefly.
- Be concise and helpful.
- Reference specific memories with dates when relevant.
- If information conflicts, prefer more recent data.
"""


class TelegramAgent:
    """Agent for the Telegram bot — builds context then uses Claude Agent SDK."""

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
        self.server = create_sdk_mcp_server(
            name="memclaw-tg", version="0.1.0", tools=tools
        )
        self.tool_names = [
            "mcp__memclaw-tg__memory_search",
            "mcp__memclaw-tg__image_search",
        ]

    def _create_tools(self):
        search = self.search
        index = self.index
        found_images = self._found_images

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
                    "content": [{
                        "type": "text",
                        "text": (
                            f"Found {len(results)} image(s):\n"
                            + "\n".join(lines)
                            + "\nImages will be sent automatically."
                        ),
                    }]
                }
            return {
                "content": [{"type": "text", "text": "No matching images found."}]
            }

        return [memory_search_tool, image_search_tool]

    async def build_context(self, question: str) -> str:
        """Build memory context to inject into the system prompt."""
        parts: list[str] = []

        # Always include permanent memory
        memory_content = self.store.read_file(self.config.memory_file)
        if memory_content.strip():
            parts.append("### Permanent Memory")
            parts.append(memory_content[:3000])

        # Semantic search for query-relevant memories
        results = await self.search.search(question, limit=10)
        if results:
            parts.append("\n### Relevant Memories")
            for r in results:
                source = Path(r.file_path).stem
                parts.append(f"[{source}] {r.content.strip()}")

        return "\n\n".join(parts) if parts else "No memories found yet."

    async def ask(self, question: str) -> tuple[str, list[dict]]:
        """Run the agent for a /ask query.

        Returns (response_text, found_images).
        """
        self._found_images.clear()

        context = await self.build_context(question)

        opts = ClaudeAgentOptions(
            system_prompt=TELEGRAM_SYSTEM_PROMPT.format(
                today=date.today().isoformat(),
                context=context,
            ),
            mcp_servers={"memclaw-tg": self.server},
            allowed_tools=self.tool_names,
            permission_mode="bypassPermissions",
            max_turns=10,
        )

        last_text = ""
        async for message in query(prompt=question, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text"):
                        last_text = block.text
            elif isinstance(message, ResultMessage) and hasattr(message, "result"):
                last_text = message.result or last_text

        return (
            last_text or "I couldn't generate a response.",
            list(self._found_images),
        )
