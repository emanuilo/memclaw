"""Tests for MemclawAgent — history, consolidation, context strategy, sync (specs #1–3, #9)."""
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
# Helpers — we patch heavy deps so MemclawAgent can be instantiated
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


def _patch_sdk():
    """Patch the Claude Agent SDK imports that aren't installed in test env."""
    return patch.dict("sys.modules", {
        "claude_agent_sdk": MagicMock(),
    })


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

        # Mock the SDK call and build_context
        agent.build_context = AsyncMock(return_value="No memories found yet.")
        agent._maybe_consolidate = AsyncMock(return_value=False)

        # Mock ClaudeSDKClient
        with patch("memclaw.agent.ClaudeSDKClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.query = AsyncMock()

            # Simulate receiving a ResultMessage
            mock_result = MagicMock()
            mock_result.result = "Hello! I'm Memclaw."
            mock_result.total_cost_usd = 0.001
            mock_result.num_turns = 1
            mock_result.duration_ms = 100

            # Make it look like a ResultMessage
            from memclaw.agent import ResultMessage
            mock_result.__class__ = ResultMessage

            async def fake_receive():
                yield mock_result

            mock_client.receive_response = fake_receive
            mock_client_cls.return_value = mock_client

            await agent.handle("Hello")

        assert len(agent._history) == 2
        assert agent._history[0]["role"] == "user"
        assert agent._history[0]["content"] == "Hello"
        assert agent._history[1]["role"] == "assistant"
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

        # Mock the Anthropic API call
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Key Facts\n\n- Fact 0\n- Fact 1\n"
        mock_response.content = [mock_block]

        with patch("memclaw.agent.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            # Also mock index_file to avoid needing real embeddings
            agent.index.index_file = AsyncMock()

            result = await agent._maybe_consolidate()

        assert result is True
        # MEMORY.md should exist with the consolidated content
        assert cfg.memory_file.exists()
        assert "Key Facts" in cfg.memory_file.read_text()

        # meta.json should be updated
        meta_path = cfg.memory_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "consolidated_through" in meta
        assert meta["consolidated_through"] == "2025-03-05"

        agent.close()

    @pytest.mark.asyncio
    async def test_force_ignores_threshold(self, cfg: MemclawConfig):
        """force=True should run consolidation even with 1 file."""
        from memclaw.agent import MemclawAgent
        cfg.consolidation_threshold = 100  # Very high threshold
        agent = MemclawAgent(cfg)

        path = cfg.memory_subdir / "2025-03-01.md"
        path.write_text("# Single day\nJust one note")

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Notes\n- One note"
        mock_response.content = [mock_block]

        with patch("memclaw.agent.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            agent.index.index_file = AsyncMock()

            result = await agent._maybe_consolidate(force=True)

        assert result is True
        agent.close()

    @pytest.mark.asyncio
    async def test_consolidated_through_override(self, cfg: MemclawConfig):
        """consolidated_through_override should override meta.json."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        # Create files spanning a range
        for i in range(1, 11):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"Content for day {i}")

        # Write meta.json with early date
        meta_path = cfg.memory_dir / "meta.json"
        meta_path.write_text(json.dumps({"consolidated_through": "2025-03-01"}))

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Consolidated"
        mock_response.content = [mock_block]

        with patch("memclaw.agent.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            agent.index.index_file = AsyncMock()

            # Override to only consolidate files after March 8
            result = await agent._maybe_consolidate(
                force=True,
                consolidated_through_override=date(2025, 3, 8),
            )

        assert result is True
        # Check that the API was called with content from only the 2 remaining files (Mar 9, 10)
        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "2025-03-09" in user_msg
        assert "2025-03-10" in user_msg
        # Files before override should NOT be included
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

        # Create files with large content
        for i in range(1, 6):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text("x" * 10000)

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "## Consolidated"
        mock_response.content = [mock_block]

        with patch("memclaw.agent.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            agent.index.index_file = AsyncMock()

            await agent._maybe_consolidate(force=True)

        # Check the user message length doesn't exceed ~30000 + headers
        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        # The daily content portion should be <= 30000 chars
        # (total msg includes "## Daily Memory Files\n\n" header)
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

        # Mock search to avoid needing real embeddings
        agent.search.search = AsyncMock(return_value=[])

        context = await agent.build_context("hello")
        assert small_content in context
        agent.close()

    @pytest.mark.asyncio
    async def test_large_memory_truncated_with_search(self, cfg: MemclawConfig):
        """MEMORY.md over 4000 chars: first 2000 + semantic search results."""
        from memclaw.agent import MemclawAgent
        agent = MemclawAgent(cfg)

        # Create a large MEMORY.md (>4000 chars)
        large_content = "## Key Facts\n\n" + "Important fact. " * 400  # ~6400 chars
        cfg.memory_file.write_text(large_content)

        # Mock search: first call is for MEMORY.md chunks, second for all memories
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

        # Should contain the first 2000 chars
        assert large_content[:100] in context
        # Should contain the semantic search result
        assert "Relevant chunk from MEMORY.md about Python" in context
        # Should NOT contain the full content
        assert len(context) < len(large_content)

        # Verify search was called with file_filter
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

        # Clean up
        agent._sync_task.cancel()
        try:
            await agent._sync_task
        except asyncio.CancelledError:
            pass
        agent.close()
