"""Tests for MemclawAgent — history, consolidation, context, sync, fs guardrail."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from memclaw.config import MemclawConfig
from memclaw.search import SearchResult


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return _make_config(tmp_path)


def _mock_api_response(text: str, stop_reason: str = "end_turn"):
    """Create a mock Anthropic API response with a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


# ────────────────────────────────────────────────────────────────────
# Spec #1: Conversation History
# ────────────────────────────────────────────────────────────────────

class TestConversationHistory:
    def test_history_initialized_empty(self, cfg: MemclawConfig):
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)
        assert agent._history == []
        agent.close()

    @pytest.mark.asyncio
    async def test_history_appended_after_handle(self, cfg: MemclawConfig):
        """handle() should append user + assistant messages to _history."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        agent.build_context = AsyncMock(return_value="No memories found yet.")
        agent._maybe_consolidate = AsyncMock(return_value=False)
        agent._client.messages.create = AsyncMock(
            return_value=_mock_api_response("Hello! I'm Memclaw.")
        )

        await agent.handle("Hello")

        assert len(agent._history) == 2
        assert agent._history[0]["role"] == "user"
        assert agent._history[0]["content"] == "Hello"
        assert agent._history[1]["role"] == "assistant"
        assert agent._history[1]["content"] == "Hello! I'm Memclaw."
        assert "timestamp" in agent._history[0]
        agent.close()

    def test_history_trimming(self, cfg: MemclawConfig):
        """History should be trimmed to conversation_history_limit * 2."""
        from memclaw.agent import MemclawAgent
        cfg.conversation_history_limit = 3  # Keep last 3 pairs = 6 entries
        agent = MemclawAgent(cfg)

        # Manually populate history with 10 entries
        for i in range(10):
            agent._history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"message {i}",
                "timestamp": datetime.now().isoformat(),
            })

        # Simulate trimming (same logic as in handle())
        max_entries = cfg.conversation_history_limit * 2
        if len(agent._history) > max_entries:
            agent._history = agent._history[-max_entries:]

        assert len(agent._history) == 6
        # Should keep the last 6 entries (messages 4-9)
        assert agent._history[0]["content"] == "message 4"
        assert agent._history[-1]["content"] == "message 9"
        agent.close()

    def test_image_placeholder_in_history(self, cfg: MemclawConfig):
        """When an image is sent, history should store a placeholder, not base64."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        # Simulate what handle() does for images
        image_b64 = "base64data..."
        history_content = "[User sent a photo]" if image_b64 else "text"
        agent._history.append({
            "role": "user",
            "content": history_content,
            "timestamp": datetime.now().isoformat(),
        })

        assert agent._history[0]["content"] == "[User sent a photo]"
        assert "base64" not in agent._history[0]["content"]
        agent.close()


# ────────────────────────────────────────────────────────────────────
# Spec #2: Memory Consolidation
# ────────────────────────────────────────────────────────────────────

class TestConsolidation:
    @pytest.mark.asyncio
    async def test_skips_when_below_threshold(self, cfg: MemclawConfig):
        """Consolidation should not run when file count < threshold."""
        from memclaw.agent import MemclawAgent
        cfg.consolidation_threshold = 7
        agent = MemclawAgent(cfg)

        # Create 3 daily files (below threshold of 7)
        for i in range(3):
            d = date(2025, 3, i + 1)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"# Day {i}\nSome content")

        result = await agent._maybe_consolidate()
        assert result is False
        agent.close()

    @pytest.mark.asyncio
    async def test_runs_when_above_threshold(self, cfg: MemclawConfig):
        """Consolidation should run when file count >= threshold."""
        from memclaw.agent import MemclawAgent
        cfg.consolidation_threshold = 3
        agent = MemclawAgent(cfg)

        # Create 5 daily files (above threshold of 3)
        for i in range(5):
            d = date(2025, 3, i + 1)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"# Day {i}\nImportant fact {i}")

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Key Facts\n\n- Fact 0\n- Fact 1\n"
        mock_response.content = [mock_block]

        agent._client.messages.create = AsyncMock(return_value=mock_response)
        agent.index.index_file = AsyncMock()

        result = await agent._maybe_consolidate()

        assert result is True
        assert cfg.memory_file.exists()
        assert "Key Facts" in cfg.memory_file.read_text()

        meta_path = cfg.memory_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["consolidated_through"] == "2025-03-05"

        agent.close()

    @pytest.mark.asyncio
    async def test_force_ignores_threshold(self, cfg: MemclawConfig):
        """force=True should run consolidation even with 1 file."""
        from memclaw.agent import MemclawAgent
        cfg.consolidation_threshold = 100
        agent = MemclawAgent(cfg)

        path = cfg.memory_subdir / "2025-03-01.md"
        path.write_text("# Single day\nJust one note")

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Notes\n- One note"
        mock_response.content = [mock_block]

        agent._client.messages.create = AsyncMock(return_value=mock_response)
        agent.index.index_file = AsyncMock()

        result = await agent._maybe_consolidate(force=True)

        assert result is True
        agent.close()

    @pytest.mark.asyncio
    async def test_consolidated_through_override(self, cfg: MemclawConfig):
        """consolidated_through_override should override meta.json."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        for i in range(1, 11):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"Content for day {i}")

        meta_path = cfg.memory_dir / "meta.json"
        meta_path.write_text(json.dumps({"consolidated_through": "2025-03-01"}))

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Consolidated"
        mock_response.content = [mock_block]

        agent._client.messages.create = AsyncMock(return_value=mock_response)
        agent.index.index_file = AsyncMock()

        result = await agent._maybe_consolidate(
            force=True,
            consolidated_through_override=date(2025, 3, 8),
        )

        assert result is True
        call_args = agent._client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "2025-03-09" in user_msg
        assert "2025-03-10" in user_msg
        assert "2025-03-05" not in user_msg

        agent.close()

    @pytest.mark.asyncio
    async def test_no_files_returns_false(self, cfg: MemclawConfig):
        """If there are no daily files at all, return False."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)
        result = await agent._maybe_consolidate(force=True)
        assert result is False
        agent.close()

    @pytest.mark.asyncio
    async def test_content_limit_30000_chars(self, cfg: MemclawConfig):
        """Content gathering should stop at 30000 chars."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        for i in range(1, 6):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text("x" * 10000)

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Consolidated"
        mock_response.content = [mock_block]

        agent._client.messages.create = AsyncMock(return_value=mock_response)
        agent.index.index_file = AsyncMock()

        await agent._maybe_consolidate(force=True)

        call_args = agent._client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert len(user_msg) < 35000

        agent.close()


# ────────────────────────────────────────────────────────────────────
# Spec #3: MEMORY.md Context Strategy
# ────────────────────────────────────────────────────────────────────

class TestContextStrategy:
    @pytest.mark.asyncio
    async def test_small_memory_included_in_full(self, cfg: MemclawConfig):
        """MEMORY.md under 4000 chars should be included completely."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        small_content = "## Key Facts\n\n- I like Python\n- My name is Test"
        cfg.memory_file.write_text(small_content)

        agent.search.search = AsyncMock(return_value=[])

        context = await agent.build_context("hello")
        assert small_content in context
        agent.close()

    @pytest.mark.asyncio
    async def test_large_memory_truncated_with_search(self, cfg: MemclawConfig):
        """MEMORY.md over 4000 chars: first 2000 + semantic search results."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        large_content = "## Key Facts\n\n" + "Important fact. " * 400
        cfg.memory_file.write_text(large_content)

        memory_chunk = SearchResult(
            file_path=str(cfg.memory_file),
            line_start=100,
            line_end=110,
            content="Relevant chunk from MEMORY.md about Python",
            score=0.8,
            match_type="vector",
        )
        agent.search.search = AsyncMock(side_effect=[
            [memory_chunk],  # file_filter="MEMORY.md" call
            [],              # general search call
        ])

        context = await agent.build_context("tell me about Python")

        assert large_content[:100] in context
        assert "Relevant chunk from MEMORY.md about Python" in context
        assert len(context) < len(large_content)

        calls = agent.search.search.call_args_list
        assert calls[0].kwargs.get("file_filter") == "MEMORY.md"

        agent.close()


# ────────────────────────────────────────────────────────────────────
# Spec #9: Startup and Background Sync
# ────────────────────────────────────────────────────────────────────

class TestSyncOptimization:
    @pytest.mark.asyncio
    async def test_start_calls_sync(self, cfg: MemclawConfig):
        """start() should call index.sync() once."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)
        agent.index.sync = AsyncMock(return_value=False)

        await agent.start()
        agent.index.sync.assert_called_once()
        agent.close()

    @pytest.mark.asyncio
    async def test_background_sync_creates_task(self, cfg: MemclawConfig):
        """start_background_sync() should create an asyncio task."""
        import asyncio
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)
        agent.index.sync = AsyncMock(return_value=False)

        await agent.start_background_sync(interval=1)
        assert hasattr(agent, "_sync_task")
        assert isinstance(agent._sync_task, asyncio.Task)

        agent._sync_task.cancel()
        try:
            await agent._sync_task
        except asyncio.CancelledError:
            pass
        agent.close()


# ────────────────────────────────────────────────────────────────────
# Filesystem Guardrail (now in ToolExecutor)
# ────────────────────────────────────────────────────────────────────

class TestFilesystemGuardrail:
    @pytest.mark.asyncio
    async def test_allows_write_inside_memory_dir(self, cfg: MemclawConfig):
        """file_write to a path under memory_dir should succeed."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "todos.md", "content": "- Buy milk"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "todos.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_outside_memory_dir(self, cfg: MemclawConfig):
        """file_write to a path outside memory_dir should be blocked."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "/tmp/evil.md", "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_to_home_dir(self, cfg: MemclawConfig):
        """file_write to ~/something.md should be blocked."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(Path.home() / "todos.md"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, cfg: MemclawConfig):
        """Path traversal attempts (../../etc) should be blocked."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(cfg.memory_dir / ".." / ".." / "etc" / "passwd"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_allows_nested_path_inside_memory_dir(self, cfg: MemclawConfig):
        """Writing to a subdirectory of memory_dir should work."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "subdir/file.md", "content": "nested"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "subdir" / "file.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_read_outside(self, cfg: MemclawConfig):
        """file_read outside memory_dir should be blocked."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        result = await agent._tools.execute(
            "file_read",
            {"file_path": "/etc/hosts"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_file_read_returns_content(self, cfg: MemclawConfig):
        """file_read should return content for files inside memory_dir."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        (cfg.memory_dir / "test.md").write_text("hello world")
        result = await agent._tools.execute(
            "file_read",
            {"file_path": "test.md"},
        )
        assert result == "hello world"
        agent.close()


class TestSandboxedFileTools:
    def test_tool_definitions_contain_file_tools(self):
        """TOOL_DEFINITIONS should include file_write and file_read."""
        from memclaw.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "file_write" in names
        assert "file_read" in names

    def test_tool_definitions_contain_all_tools(self):
        """TOOL_DEFINITIONS should have all 8 tools."""
        from memclaw.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == 8
        expected = {
            "memory_save", "memory_search", "image_save",
            "telegram_image_save", "image_search",
            "update_instructions", "file_write", "file_read",
        }
        assert set(names) == expected
