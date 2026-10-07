"""Tests for MemclawAgent — sessions, consolidation, context, sync, fs guardrail.

These tests run against a backend-agnostic `FakeBackend` so they exercise the
orchestration in `MemclawAgent` without depending on any SDK. Per-backend
behavior (env scrubbing, MCP wrapping, …) is tested separately in
`tests/backends/`.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
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
    reset_calls: list[str] = field(default_factory=list)
    # Session id each turn reports; None mimics a backend without sessions.
    session_id: str | None = "sess-1"
    turn_error: Exception | None = None

    async def on_agent_start(self, tool_executor) -> None:
        pass

    async def on_agent_shutdown(self) -> None:
        pass

    async def run_one_shot(self, *, system_prompt: str, user_message: str) -> str:
        self.one_shot_calls.append({"system": system_prompt, "user": user_message})
        return self.one_shot_response

    async def run_turn(self, **kwargs) -> TurnResult:
        self.turn_calls.append(kwargs)
        if self.turn_error is not None:
            raise self.turn_error
        result = TurnResult(**vars(self.turn_response))
        result.session_id = self.session_id
        return result

    async def reset_session(self, session_key: str) -> None:
        self.reset_calls.append(session_key)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def _make_agent(cfg: MemclawConfig, backend: FakeBackend, platform: str | None = None):
    from memclaw.agent import MemclawAgent
    return MemclawAgent(cfg, platform=platform, backend=backend)


def _chat_agent(cfg: MemclawConfig, backend: FakeBackend, platform: str | None = None):
    """An agent ready for handle(): no search, no consolidation."""
    agent = _make_agent(cfg, backend, platform)
    agent.search.search = AsyncMock(return_value=[])
    agent._maybe_consolidate = AsyncMock(return_value=False)
    return agent


# ────────────────────────────────────────────────────────────────────
# Conversation sessions
# ────────────────────────────────────────────────────────────────────

class TestSessions:
    @pytest.mark.asyncio
    async def test_first_turn_starts_a_session(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        fake_backend.turn_response = TurnResult(text="Hello! I'm Memclaw.")
        agent = _chat_agent(cfg, fake_backend)

        reply, _ = await agent.handle("Hello")

        assert reply == "Hello! I'm Memclaw."
        call = fake_backend.turn_calls[0]
        assert call["user_message"] == "Hello"
        assert call["session_key"] == "cli"
        assert call["resume_session_id"] is None
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert stored["cli"]["session_id"] == "sess-1"
        agent.close()

    @pytest.mark.asyncio
    async def test_session_key_is_per_chat_and_platform(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        await agent.handle("hi", chat_id="42")
        assert fake_backend.turn_calls[0]["session_key"] == "telegram:42"
        agent.close()

    @pytest.mark.asyncio
    async def test_next_turn_and_restart_resume_the_session(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        await agent.handle("two")
        assert fake_backend.turn_calls[1]["resume_session_id"] == "sess-1"
        agent.close()

        restarted = _chat_agent(cfg, fake_backend)
        await restarted.handle("three")
        assert fake_backend.turn_calls[2]["resume_session_id"] == "sess-1"
        restarted.close()

    @pytest.mark.asyncio
    async def test_new_session_id_after_a_failed_resume_is_stored(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        fake_backend.session_id = "sess-2"  # backend couldn't resume, started fresh
        await agent.handle("two")
        await agent.handle("three")
        assert fake_backend.turn_calls[2]["resume_session_id"] == "sess-2"
        agent.close()

    @pytest.mark.asyncio
    async def test_session_from_other_instructions_is_not_resumed(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        (cfg.memory_dir / "sessions.json").write_text(json.dumps({
            "cli": {"session_id": "old", "fingerprint": "stale",
                    "agents_hash": "x", "memory_hash": "y"},
        }))
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("hi")

        call = fake_backend.turn_calls[0]
        assert call["resume_session_id"] is None
        # A fresh session gets the files in its system prompt, not as updates.
        assert "Updated AGENTS.md" not in call["context"]
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert stored["cli"]["session_id"] == "sess-1"
        agent.close()

    @pytest.mark.asyncio
    async def test_backend_without_sessions_stores_nothing(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        fake_backend.session_id = None
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("hi")
        assert not (cfg.memory_dir / "sessions.json").exists()
        agent.close()

    @pytest.mark.asyncio
    async def test_reset_conversation(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        await agent.handle("one", chat_id="42")
        await agent.handle("other chat", chat_id="7")
        agent.record_reminder_fired("⏰ Reminder: stretch", "42")

        await agent.reset_conversation("42")

        assert fake_backend.reset_calls == ["telegram:42"]
        stored = json.loads((cfg.memory_dir / "sessions.json").read_text())
        assert "telegram:42" not in stored
        assert "telegram:7" in stored
        await agent.handle("two", chat_id="42")
        call = fake_backend.turn_calls[-1]
        assert call["resume_session_id"] is None
        assert "stretch" not in call["context"]
        agent.close()


class TestSystemPromptAndContext:
    @pytest.mark.asyncio
    async def test_system_prompt_is_static(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        cfg.agent_file.write_text("Be terse.")
        cfg.memory_file.write_text("## Key Facts\n- Name: Test")
        agent = _chat_agent(cfg, fake_backend)

        await agent.handle("one")
        await agent.handle("two")

        first, second = (c["system_prompt"] for c in fake_backend.turn_calls)
        assert first == second
        assert "Be terse." in first
        assert "- Name: Test" in first
        assert "Current local time" not in first
        agent.close()

    @pytest.mark.asyncio
    async def test_context_block_carries_time_and_memories(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        agent.search.search = AsyncMock(return_value=[
            SearchResult(file_path="memory/2025-03-01.md", line_start=0, line_end=1,
                         content="Bought   a red\nbike", score=0.9, match_type="hybrid"),
            SearchResult(file_path=str(cfg.memory_file), line_start=0, line_end=1,
                         content="already in the system prompt", score=0.8,
                         match_type="hybrid"),
            SearchResult(file_path="memory/2025-03-02.md", line_start=0, line_end=1,
                         content="x" * 2000, score=0.7, match_type="hybrid"),
        ])

        await agent.handle("what bike?")

        context = fake_backend.turn_calls[0]["context"]
        assert context.startswith("<context>\nCurrent local time: ")
        assert context.endswith("</context>")
        assert "- [2025-03-01] Bought a red bike" in context
        assert "already in the system prompt" not in context
        assert "x" * 501 not in context  # long snippets are cut
        agent.close()

    @pytest.mark.asyncio
    async def test_changed_agents_md_is_sent_once(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        cfg.agent_file.write_text("Be terse.")
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        assert "Updated AGENTS.md" not in fake_backend.turn_calls[0]["context"]

        cfg.agent_file.write_text("Be terse.\nAlways answer in Spanish.")
        await agent.handle("two")
        await agent.handle("three")

        second = fake_backend.turn_calls[1]["context"]
        assert "Updated AGENTS.md" in second
        assert "Always answer in Spanish." in second
        assert "Updated AGENTS.md" not in fake_backend.turn_calls[2]["context"]
        agent.close()

    @pytest.mark.asyncio
    async def test_changed_memory_md_is_sent_once_even_after_restart(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)
        await agent.handle("one")
        agent.close()

        cfg.memory_file.write_text("## Key Facts\n- New fact")
        restarted = _chat_agent(cfg, fake_backend)
        await restarted.handle("two")
        await restarted.handle("three")

        second = fake_backend.turn_calls[1]["context"]
        assert "Updated permanent memory" in second
        assert "- New fact" in second
        assert "Updated permanent memory" not in fake_backend.turn_calls[2]["context"]
        restarted.close()

    @pytest.mark.asyncio
    async def test_own_writes_during_a_turn_are_not_sent_back(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)

        async def turn_that_saves(**kwargs):
            # memory_save(permanent=true) / update_instructions mid-turn.
            cfg.memory_file.write_text("## Key Facts\n- Old fact\n- Dog is Bruno")
            cfg.agent_file.write_text("Be terse.")
            return TurnResult(text="saved", session_id="sess-1")

        agent.backend.run_turn = turn_that_saves
        await agent.handle("my dog is Bruno")
        agent.backend.run_turn = FakeBackend.run_turn.__get__(fake_backend)
        await agent.handle("next")

        context = fake_backend.turn_calls[-1]["context"]
        assert "Updated permanent memory" not in context
        assert "Updated AGENTS.md" not in context
        agent.close()

    @pytest.mark.asyncio
    async def test_consolidation_during_a_turn_is_still_sent(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        cfg.memory_file.write_text("## Key Facts\n- Old fact")
        agent = _chat_agent(cfg, fake_backend)

        async def turn_racing_consolidation(**kwargs):
            cfg.memory_file.write_text("## Key Facts\n- Consolidated")
            agent._memory_rewrites += 1
            return TurnResult(text="ok", session_id="sess-1")

        agent.backend.run_turn = turn_racing_consolidation
        await agent.handle("one")
        agent.backend.run_turn = FakeBackend.run_turn.__get__(fake_backend)
        await agent.handle("two")

        context = fake_backend.turn_calls[-1]["context"]
        assert "Updated permanent memory" in context
        assert "- Consolidated" in context
        agent.close()

    @pytest.mark.asyncio
    async def test_reminder_notes_go_into_the_next_message_of_that_chat(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        agent.record_reminder_fired("⏰ Reminder: call Alex", "42")

        await agent.handle("other chat", chat_id="7")
        await agent.handle("done!", chat_id="42")
        await agent.handle("and now?", chat_id="42")

        other, first, second = (c["context"] for c in fake_backend.turn_calls)
        assert "call Alex" not in other
        assert "- ⏰ Reminder: call Alex" in first
        assert "call Alex" not in second
        agent.close()

    @pytest.mark.asyncio
    async def test_reminder_note_survives_a_failed_turn(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend, platform="telegram")
        agent.record_reminder_fired("⏰ Reminder: call Alex", "42")
        fake_backend.turn_error = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await agent.handle("hi", chat_id="42")
        fake_backend.turn_error = None
        await agent.handle("hi again", chat_id="42")
        assert "call Alex" in fake_backend.turn_calls[-1]["context"]
        agent.close()


class TestBackgroundConsolidation:
    @pytest.mark.asyncio
    async def test_runs_after_the_reply_without_blocking_it(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_consolidate(**kwargs):
            started.set()
            await release.wait()
            return True

        agent._maybe_consolidate = AsyncMock(side_effect=slow_consolidate)

        await agent.handle("hi")  # returns while consolidation is still pending
        await asyncio.wait_for(started.wait(), 1)
        await agent.handle("again")  # one already running: no second one
        release.set()
        await agent._consolidation_task
        assert agent._maybe_consolidate.await_count == 1
        agent.close()

    @pytest.mark.asyncio
    async def test_failure_is_logged_not_raised(self, cfg: MemclawConfig, fake_backend: FakeBackend):
        agent = _chat_agent(cfg, fake_backend)
        agent._maybe_consolidate = AsyncMock(side_effect=RuntimeError("API down"))
        reply, _ = await agent.handle("hi")
        await agent._consolidation_task  # does not raise
        assert reply
        agent.close()

    @pytest.mark.asyncio
    async def test_aclose_cancels_a_running_consolidation(
        self, cfg: MemclawConfig, fake_backend: FakeBackend,
    ):
        agent = _chat_agent(cfg, fake_backend)
        async def forever(**kwargs):
            await asyncio.sleep(60)

        agent._maybe_consolidate = AsyncMock(side_effect=forever)
        await agent.handle("hi")
        await asyncio.sleep(0)
        task = agent._consolidation_task
        await agent.aclose()
        assert task.cancelled()


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
# Spec #3: MEMORY.md in the system prompt
# ────────────────────────────────────────────────────────────────────

class TestPermanentMemoryPrompt:
    def test_small_memory_included_in_full(self, cfg: MemclawConfig):
        from memclaw.agent import _load_permanent_memory

        cfg.memory_file.write_text("## Key Facts\n\n- I like Python")
        assert _load_permanent_memory(cfg) == "## Key Facts\n\n- I like Python"

    def test_runaway_memory_is_truncated(self, cfg: MemclawConfig):
        from memclaw.agent import _MEMORY_PROMPT_CHARS, _load_permanent_memory

        cfg.memory_file.write_text("x" * (_MEMORY_PROMPT_CHARS + 100))
        text = _load_permanent_memory(cfg)
        assert len(text) < _MEMORY_PROMPT_CHARS + 100
        assert "truncated" in text

    def test_empty_memory(self, cfg: MemclawConfig):
        from memclaw.agent import _load_permanent_memory

        cfg.memory_file.write_text("")
        assert _load_permanent_memory(cfg) == "(empty)"


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
