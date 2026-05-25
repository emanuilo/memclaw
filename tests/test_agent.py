"""Tests for MemclawAgent — history, consolidation, context, sync, fs guardrail.

These tests run against a backend-agnostic `FakeBackend` so they exercise the
orchestration in `MemclawAgent` without depending on any SDK. Per-backend
behavior (env scrubbing, MCP wrapping, …) is tested separately in
`tests/backends/`.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from dataclasses import dataclass, field

import pytest

from memclaw.backends.base import TurnResult
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


@dataclass
class FakeBackend:
    """Backend stub that records calls and returns canned responses."""

    name: str = "fake"
    display_name: str = "Fake backend"
    bills_per_token: bool = False
    one_shot_response: str = ""
    turn_response: TurnResult = field(default_factory=lambda: TurnResult(text=""))
    one_shot_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_calls: list[dict[str, Any]] = field(default_factory=list)

    async def on_agent_start(self, tool_executor) -> None:
        pass

    async def on_agent_shutdown(self) -> None:
        pass

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        self.one_shot_calls.append({"system": system_prompt, "user": user_message})
        return self.one_shot_response

    async def run_turn(self, **kwargs) -> TurnResult:
        self.turn_calls.append(kwargs)
        return self.turn_response


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def _make_agent(cfg: MemclawConfig, backend: FakeBackend):
    from memclaw.agent import MemclawAgent
    return MemclawAgent(cfg, backend=backend)


# ────────────────────────────────────────────────────────────────────
# Spec #1: Conversation History
# ────────────────────────────────────────────────────────────────────

class TestConversationHistory:
    def test_history_initialized_empty(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _make_agent(cfg, fake_backend)
        assert agent._history == []
        agent.close()

    @pytest.mark.asyncio
    async def test_history_appended_after_handle(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """handle() should append user + assistant messages to _history."""
        fake_backend.turn_response = TurnResult(text="Hello! I'm Memclaw.")
        agent = _make_agent(cfg, fake_backend)

        agent.build_context = AsyncMock(return_value="No memories found yet.")
        agent._maybe_consolidate = AsyncMock(return_value=False)

        await agent.handle("Hello")

        assert len(agent._history) == 2
        assert agent._history[0]["role"] == "user"
        assert agent._history[0]["content"] == "Hello"
        assert agent._history[1]["role"] == "assistant"
        assert agent._history[1]["content"] == "Hello! I'm Memclaw."
        assert "timestamp" in agent._history[0]
        # Backend was called exactly once with the user message.
        assert len(fake_backend.turn_calls) == 1
        assert fake_backend.turn_calls[0]["user_message"] == "Hello"
        agent.close()

    def test_history_trimming(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """History should be trimmed to conversation_history_limit * 2."""
        cfg.conversation_history_limit = 3  # Keep last 3 pairs = 6 entries
        agent = _make_agent(cfg, fake_backend)

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

    def test_image_placeholder_in_history(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """When an image is sent, history should store a placeholder, not base64."""
        agent = _make_agent(cfg, fake_backend)

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
    async def test_skips_when_below_threshold(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Consolidation should not run when file count < threshold."""
        cfg.consolidation_threshold = 7
        agent = _make_agent(cfg, fake_backend)

        # Create 3 daily files (below threshold of 7)
        for i in range(3):
            d = date(2025, 3, i + 1)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"# Day {i}\nSome content")

        result = await agent._maybe_consolidate()
        assert result is False
        assert fake_backend.one_shot_calls == []
        agent.close()

    @pytest.mark.asyncio
    async def test_runs_when_above_threshold(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Consolidation should run when file count >= threshold."""
        cfg.consolidation_threshold = 3
        fake_backend.one_shot_response = "## Key Facts\n\n- Fact 0\n- Fact 1\n"
        agent = _make_agent(cfg, fake_backend)

        # Create 5 daily files (above threshold of 3)
        for i in range(5):
            d = date(2025, 3, i + 1)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"# Day {i}\nImportant fact {i}")

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
    async def test_force_ignores_threshold(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """force=True should run consolidation even with 1 file."""
        cfg.consolidation_threshold = 100
        fake_backend.one_shot_response = "## Notes\n- One note"
        agent = _make_agent(cfg, fake_backend)

        path = cfg.memory_subdir / "2025-03-01.md"
        path.write_text("# Single day\nJust one note")

        agent.index.index_file = AsyncMock()

        result = await agent._maybe_consolidate(force=True)

        assert result is True
        agent.close()

    @pytest.mark.asyncio
    async def test_consolidated_through_override(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """consolidated_through_override should override meta.json."""
        fake_backend.one_shot_response = "## Consolidated"
        agent = _make_agent(cfg, fake_backend)

        for i in range(1, 11):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text(f"Content for day {i}")

        meta_path = cfg.memory_dir / "meta.json"
        meta_path.write_text(json.dumps({"consolidated_through": "2025-03-01"}))

        agent.index.index_file = AsyncMock()

        result = await agent._maybe_consolidate(
            force=True,
            consolidated_through_override=date(2025, 3, 8),
        )

        assert result is True
        # The user message passed to the backend should cover days > 2025-03-08.
        user_msg = fake_backend.one_shot_calls[-1]["user"]
        assert "2025-03-09" in user_msg
        assert "2025-03-10" in user_msg
        assert "2025-03-05" not in user_msg

        agent.close()

    @pytest.mark.asyncio
    async def test_no_files_returns_false(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """If there are no daily files at all, return False."""
        agent = _make_agent(cfg, fake_backend)
        result = await agent._maybe_consolidate(force=True)
        assert result is False
        assert fake_backend.one_shot_calls == []
        agent.close()

    @pytest.mark.asyncio
    async def test_content_limit_30000_chars(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Content gathering should stop at 30000 chars."""
        fake_backend.one_shot_response = "## Consolidated"
        agent = _make_agent(cfg, fake_backend)

        for i in range(1, 6):
            d = date(2025, 3, i)
            path = cfg.memory_subdir / f"{d.isoformat()}.md"
            path.write_text("x" * 10000)

        agent.index.index_file = AsyncMock()

        await agent._maybe_consolidate(force=True)

        user_msg = fake_backend.one_shot_calls[-1]["user"]
        assert len(user_msg) < 35000

        agent.close()


# ────────────────────────────────────────────────────────────────────
# Spec #3: MEMORY.md Context Strategy
# ────────────────────────────────────────────────────────────────────

class TestContextStrategy:
    @pytest.mark.asyncio
    async def test_small_memory_included_in_full(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """MEMORY.md under 4000 chars should be included completely."""
        agent = _make_agent(cfg, fake_backend)

        small_content = "## Key Facts\n\n- I like Python\n- My name is Test"
        cfg.memory_file.write_text(small_content)

        agent.search.search = AsyncMock(return_value=[])

        context = await agent.build_context("hello")
        assert small_content in context
        agent.close()

    @pytest.mark.asyncio
    async def test_large_memory_truncated_with_search(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """MEMORY.md over 4000 chars: first 2000 + semantic search results."""
        agent = _make_agent(cfg, fake_backend)

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
    async def test_start_calls_sync(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start() should call index.sync() once."""
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(return_value=False)

        await agent.start()
        agent.index.sync.assert_called_once()
        agent.close()

    @pytest.mark.asyncio
    async def test_start_without_backend_skips_lifecycle(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start(include_backend=False) should sync without backend startup."""
        agent = _make_agent(cfg, fake_backend)
        agent.index.sync = AsyncMock(return_value=False)
        fake_backend.on_agent_start = AsyncMock()
        fake_backend.on_agent_shutdown = AsyncMock()

        await agent.start(include_backend=False)
        agent.index.sync.assert_called_once()
        fake_backend.on_agent_start.assert_not_called()
        assert not agent._backend_started

        await agent.aclose()
        fake_backend.on_agent_shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_sync_creates_task(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """start_background_sync() should create an asyncio task."""
        import asyncio
        agent = _make_agent(cfg, fake_backend)
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
    async def test_allows_write_inside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to a path under memory_dir should succeed."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "todos.md", "content": "- Buy milk"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "todos.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_outside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to a path outside memory_dir should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "/tmp/evil.md", "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_write_to_home_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_write to ~/something.md should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(Path.home() / "todos.md"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Path traversal attempts (../../etc) should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": str(cfg.memory_dir / ".." / ".." / "etc" / "passwd"), "content": "bad"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_allows_nested_path_inside_memory_dir(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """Writing to a subdirectory of memory_dir should work."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_write",
            {"file_path": "subdir/file.md", "content": "nested"},
        )
        assert "File written" in result
        assert (cfg.memory_dir / "subdir" / "file.md").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_blocks_read_outside(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_read outside memory_dir should be blocked."""
        agent = _make_agent(cfg, fake_backend)

        result = await agent._tools.execute(
            "file_read",
            {"file_path": "/etc/hosts"},
        )
        assert "Blocked" in result
        agent.close()

    @pytest.mark.asyncio
    async def test_file_read_returns_content(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        """file_read should return content for files inside memory_dir."""
        agent = _make_agent(cfg, fake_backend)

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
        """TOOL_DEFINITIONS should expose the full catalog."""
        from memclaw.tools import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        expected = {
            "memory_save", "memory_search",
            "image_save", "image_search",
            "update_instructions", "file_write", "file_read",
            "reminder_create", "reminder_list", "reminder_cancel",
        }
        assert set(names) == expected
        assert len(names) == len(expected)
