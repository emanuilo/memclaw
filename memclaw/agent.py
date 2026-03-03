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
