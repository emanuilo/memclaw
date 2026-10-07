"""Tests for the Cursor SDK backend."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from cursor_sdk import (
    HttpMcpServerConfig,
    ModelParameterDefinition,
    ModelParameterDefinitionValue,
    SDKModel,
)

from memclaw.backends.base import TurnResult
from memclaw.backends import REGISTRY, get_backend_class
from memclaw.backends.cursor import (
    CursorAgentBackend,
    _agent_options,
    _build_combined_prompt,
    _build_user_message,
)
from memclaw.backends.cursor_sdk_adapter import (
    RunUsageTracker,
    accumulate_usage,
    assistant_message_text,
    collect_run_result,
    normalize_tool_call,
    parse_cursor_usage,
    record_interaction_usage,
)
from memclaw.backends.mcp_bridge import HttpMcpServer
from memclaw.backends.cursor_hooks import cursor_hooks_installed
from memclaw.config import MemclawConfig


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Prevent shell env from leaking into MemclawConfig."""
    for name in ("CURSOR_API_KEY", "CURSOR_MODEL", "CURSOR_EFFORT"):
        monkeypatch.delenv(name, raising=False)


def _make_config(
    tmp_path: Path,
    *,
    cursor_api_key: str = "",
    cursor_model: str = "",
    cursor_effort: str = "",
) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        cursor_api_key=cursor_api_key,
        cursor_model=cursor_model,
        cursor_effort=cursor_effort,
    )


class TestRegistry:
    def test_both_backends_registered(self):
        assert "cursor" in REGISTRY
        assert "claude" in REGISTRY
        assert REGISTRY["cursor"] is CursorAgentBackend

    def test_get_backend_class_by_name(self):
        assert get_backend_class("cursor") is CursorAgentBackend

    def test_get_backend_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown agent backend"):
            get_backend_class("nonexistent")


class TestCursorAgentBackendConfig:
    def test_is_configured_false_without_key(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert CursorAgentBackend.is_configured(cfg) is False

    def test_is_configured_true_with_key(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        assert CursorAgentBackend.is_configured(cfg) is True

    def test_configuration_help_mentions_env(self):
        help_text = CursorAgentBackend.configuration_help()
        assert "CURSOR_API_KEY" in help_text
        assert "AGENT_BACKEND=cursor" in help_text

    def test_init_reads_config(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            cursor_api_key="crsr_test_key",
            cursor_model="composer-2.5",
        )
        backend = CursorAgentBackend(cfg)
        assert backend._api_key == "crsr_test_key"
        assert backend._model == "composer-2.5"
        assert backend._cwd == str(cfg.memory_dir)
        assert os.environ["MEMCLAW_MEMORY_DIR"] == str(cfg.memory_dir)
        assert backend.bills_per_token is True
        assert cursor_hooks_installed(cfg.memory_dir) is False

    def test_status_rows_show_the_model(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_model="composer-3")
        assert CursorAgentBackend.status_rows(cfg) == [
            ("Model", "composer-3"), ("Effort", "default"),
        ]

    def test_status_rows_show_the_effort(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_effort="high")
        assert ("Effort", "high") in CursorAgentBackend.status_rows(cfg)

    def test_status_rows_fall_back_to_default_model(self, tmp_path):
        rows = CursorAgentBackend.status_rows(_make_config(tmp_path))
        assert rows[0] == ("Model", "composer-2.5")


class TestPromptBuilding:
    def test_agent_options_loads_project_setting_sources(self):
        options = _agent_options(
            api_key="test",
            cwd="/tmp/memclaw",
            model="composer-2.5",
        )
        assert options.local.setting_sources == ["project"]

    def test_normalize_tool_call_unwraps_mcp_wrapper(self):
        name, args = normalize_tool_call(
            "mcp",
            {
                "providerIdentifier": "memclaw",
                "toolName": "memory_save",
                "args": {"content": "hello"},
            },
        )
        assert name == "memory_save"
        assert args == {"content": "hello"}

    def test_normalize_tool_call_strips_memclaw_prefix(self):
        name, args = normalize_tool_call("memclaw_file_write", {"file_path": "x.md"})
        assert name == "file_write"
        assert args == {"file_path": "x.md"}

    def test_combined_prompt_includes_system_and_user(self):
        prompt = _build_combined_prompt(
            system_prompt="You are Memclaw.",
            user_message="Hello",
        )
        assert "You are Memclaw." in prompt
        assert "Hello" in prompt
        assert "---" in prompt

    def test_follow_up_message_has_no_system_prompt(self):
        message = _build_user_message(
            system_prompt="", context="<context>c</context>", user_message="Hi",
        )
        assert message == "<context>c</context>\n\nHi"

    def test_image_uses_sdk_image(self):
        message = _build_user_message(
            system_prompt="Sys",
            user_message="Look at this",
            image_b64="abc123",
            image_media_type="image/png",
        )
        assert message.text == _build_combined_prompt(
            system_prompt="Sys",
            user_message="Look at this",
        )
        assert len(message.images) == 1
        assert message.images[0].mime_type == "image/png"


class TestCollectRunResult:
    def test_assistant_message_text_concatenates_blocks(self):
        message = SimpleNamespace(
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Here's a motivational video for "),
                    SimpleNamespace(type="text", text="you."),
                ],
            ),
        )
        assert assistant_message_text(message) == "Here's a motivational video for you."

    def test_parse_cursor_usage_accepts_snake_and_camel_case(self):
        parsed = parse_cursor_usage(
            {
                "inputTokens": 120,
                "output_tokens": 45,
                "cacheReadTokens": 10,
                "cacheWriteTokens": 5,
                "totalCostUsd": 0.0123,
            }
        )
        assert parsed["input_tokens"] == 120
        assert parsed["output_tokens"] == 45
        assert parsed["cache_read_tokens"] == 10
        assert parsed["cache_creation_tokens"] == 5
        assert parsed["cost_usd"] == pytest.approx(0.0123)

    def test_accumulate_usage_sums_multiple_turns(self):
        totals: dict[str, int | float | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": None,
        }
        accumulate_usage(totals, {"input_tokens": 100, "output_tokens": 20})
        accumulate_usage(totals, {"input_tokens": 50, "output_tokens": 10, "total_cost_usd": 0.01})
        assert totals["input_tokens"] == 150
        assert totals["output_tokens"] == 30
        assert totals["cost_usd"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_collect_run_result_prefers_longer_wait_text(self):
        async def _events():
            yield SimpleNamespace(
                sdk_message=SimpleNamespace(
                    type="assistant",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text="you.")],
                    ),
                ),
                interaction_update=None,
                result=None,
            )

        mock_run = AsyncMock()
        mock_run.events = MagicMock(return_value=_events())
        mock_run.wait = AsyncMock(
            return_value=SimpleNamespace(
                result="Here's your motivational video: youtube.com/watch?v=abc",
                num_turns=1,
            ),
        )

        result = await collect_run_result(mock_run, max_turns=10)
        assert "motivational video" in result.text
        assert result.text != "you."

    @pytest.mark.asyncio
    async def test_collect_run_result_reports_each_tool_call_once(self):
        def _call(call_id, status, name, args):
            return SimpleNamespace(
                sdk_message=SimpleNamespace(
                    type="tool_call", status=status, name=name, args=args, call_id=call_id,
                ),
            )

        async def _events():
            yield _call("1", "running", "mcp", {"toolName": "memory_search",
                                                "args": {"query": "dog"}})
            yield _call("1", "running", "mcp", {"toolName": "memory_search",
                                                "args": {"query": "dog"}})
            yield _call("1", "completed", "mcp", {"toolName": "memory_search"})
            yield _call("2", "running", "memclaw_memory_save", {"content": "x"})

        mock_run = AsyncMock()
        mock_run.events = MagicMock(return_value=_events())
        mock_run.wait = AsyncMock(return_value=SimpleNamespace(result="ok", num_turns=1))
        steps = []

        async def on_tool(step):
            steps.append(step)

        await collect_run_result(mock_run, max_turns=10, on_tool=on_tool)
        assert [(s.index, s.name, s.summary) for s in steps] == [
            (1, "memory_search", "Searching memories: dog"),
            (2, "memory_save", "Saving to memory"),
        ]

    @pytest.mark.asyncio
    async def test_collect_run_result_reads_turn_ended_usage(self):
        tracker = RunUsageTracker()
        tracker.on_delta(
            SimpleNamespace(
                type="turn-ended",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        )

        mock_run = AsyncMock()
        mock_run.events = MagicMock(return_value=_empty_events())
        mock_run.wait = AsyncMock(return_value=SimpleNamespace(result="done", num_turns=1))

        result = await collect_run_result(
            mock_run,
            max_turns=10,
            usage_tracker=tracker,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.num_turns == 1

    def test_record_interaction_usage_falls_back_to_token_delta(self):
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "token_delta_sum": 0,
            "cost_usd": None,
            "turn_count": 0,
        }
        record_interaction_usage(totals, SimpleNamespace(type="token-delta", tokens=42))
        tracker = RunUsageTracker()
        tracker.totals = totals
        result = TurnResult(text="hi")
        tracker.apply_to_result(result)
        assert result.output_tokens == 42


_MCP = HttpMcpServerConfig(url="http://127.0.0.1:8765/mcp", type="http")


def _make_run(reply: str = "Turn response", num_turns: int = 1):
    run = AsyncMock()
    run.events = MagicMock(return_value=_empty_events())
    run.wait = AsyncMock(return_value=SimpleNamespace(result=reply, num_turns=num_turns))
    return run


def _make_agent(agent_id: str = "agent-1", replies=None):
    """A mock AsyncAgent that records each sent message."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.sent = []
    replies = iter(replies or ["Turn response"] * 10)

    async def _send(message, options):
        agent.sent.append((message, options))
        return _make_run(next(replies))

    agent.send = AsyncMock(side_effect=_send)
    agent.close = AsyncMock()
    return agent


def _make_client(*, create=None, resume=None):
    client = MagicMock()
    client.agents.create = AsyncMock(side_effect=create or [_make_agent()])
    client.agents.resume = AsyncMock(side_effect=resume or [])
    client.models.list = AsyncMock(return_value=[])
    client.aclose = AsyncMock()
    return client


@contextmanager
def _patched(backend, *clients):
    """Run *backend* against mock bridge clients (one per launch) and a stub MCP server."""
    with patch.object(
        backend, "_launch_client", new_callable=AsyncMock, side_effect=list(clients),
    ) as launch, patch.object(
        backend, "_ensure_mcp_server", new_callable=AsyncMock
    ), patch.object(
        HttpMcpServer, "config", new_callable=PropertyMock, return_value=_MCP
    ):
        yield launch


async def _turn(backend, text="User", *, key="cli", resume=None, context="<context>ctx</context>"):
    return await backend.run_turn(
        system_prompt="System", context=context, user_message=text,
        tool_executor=MagicMock(), session_key=key, resume_session_id=resume,
    )


def _backend(tmp_path, **kwargs) -> CursorAgentBackend:
    return CursorAgentBackend(_make_config(tmp_path, cursor_api_key="crsr_test_key", **kwargs))


def _sdk_model(model_id, *, effort_values=()):
    params = ()
    if effort_values:
        params = (ModelParameterDefinition(
            id="reasoning",
            values=tuple(ModelParameterDefinitionValue(value=v) for v in effort_values),
        ),)
    return SDKModel(id=model_id, display_name=model_id.title(), parameters=params)


class TestCursorAgentBackendRuns:
    @pytest.mark.asyncio
    async def test_run_one_shot(self, tmp_path):
        backend = _backend(tmp_path)
        client = _make_client()

        with _patched(backend, client), patch(
            "cursor_sdk.AsyncAgent.prompt", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.return_value = SimpleNamespace(result="Hello from Cursor")
            text = await backend.run_one_shot(system_prompt="System", user_message="User")

        assert text == "Hello from Cursor"
        mock_prompt.assert_awaited_once()
        call_prompt = mock_prompt.await_args.args[0]
        assert "System" in call_prompt
        assert "User" in call_prompt
        assert mock_prompt.await_args.kwargs["client"] is client

    @pytest.mark.asyncio
    async def test_run_one_shot_missing_key_raises(self, tmp_path):
        backend = CursorAgentBackend(_make_config(tmp_path))
        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            await backend.run_one_shot(system_prompt="S", user_message="U")

    @pytest.mark.asyncio
    async def test_launch_runs_bridge_in_memory_dir(self, tmp_path):
        """The agent store path follows the bridge's cwd, so it must be fixed."""
        backend = _backend(tmp_path)
        with patch(
            "cursor_sdk.AsyncClient.launch_bridge", new_callable=AsyncMock,
            return_value=MagicMock(_owned_bridge=None),
        ) as launch:
            await backend._launch_client()
        argv = launch.await_args.args[0]
        assert argv[:2] == ["/bin/sh", "-c"]
        assert argv[3] == backend._cwd
        assert launch.await_args.kwargs["workspace"] == backend._cwd

    @pytest.mark.asyncio
    async def test_run_turn(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent(replies=["Turn response"])
        client = _make_client(create=[agent])

        with _patched(backend, client):
            result = await _turn(backend)

        assert result.text == "Turn response"
        assert result.session_id == "agent-1"
        assert result.cost_usd is None
        create_options = client.agents.create.await_args.args[0]
        assert create_options.mcp_servers == {"memclaw": _MCP}
        assert create_options.model.id == "composer-2.5"
        _, send_options = agent.sent[0]
        assert send_options.on_delta is not None
        assert send_options.mcp_servers == {"memclaw": _MCP}
        assert send_options.model.id == "composer-2.5"
        assert send_options.local is None
        agent.close.assert_not_awaited()  # kept open for the next turn

    @pytest.mark.asyncio
    async def test_agent_is_reused_and_system_prompt_only_sent_first(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent(replies=["First", "Second"])
        client = _make_client(create=[agent])

        with _patched(backend, client) as launch:
            first = await _turn(backend, "My name is Ana")
            second = await _turn(backend, "What's my name?", resume=first.session_id)

        launch.assert_awaited_once()  # one bridge for the process
        client.agents.create.assert_awaited_once()
        client.agents.resume.assert_not_awaited()
        assert second.session_id == "agent-1"
        first_msg, second_msg = agent.sent[0][0], agent.sent[1][0]
        assert first_msg.startswith("System")
        assert "<context>ctx</context>" in first_msg and "My name is Ana" in first_msg
        assert "System" not in second_msg
        assert second_msg == "<context>ctx</context>\n\nWhat's my name?"

    @pytest.mark.asyncio
    async def test_sessions_are_separate_per_key(self, tmp_path):
        backend = _backend(tmp_path)
        a, b = _make_agent("agent-a"), _make_agent("agent-b")
        client = _make_client(create=[a, b])

        with _patched(backend, client):
            ra = await _turn(backend, key="chat-a")
            rb = await _turn(backend, key="chat-b")

        assert (ra.session_id, rb.session_id) == ("agent-a", "agent-b")

    @pytest.mark.asyncio
    async def test_resume_passes_model_and_forces_first_send(self, tmp_path):
        backend = _backend(tmp_path, cursor_model="gpt-6")
        agent = _make_agent("agent-old", replies=["One", "Two"])
        client = _make_client(resume=[agent])

        with _patched(backend, client):
            result = await _turn(backend, resume="agent-old")
            await _turn(backend, resume="agent-old")

        client.agents.create.assert_not_awaited()
        agent_id, options = client.agents.resume.await_args.args
        assert agent_id == "agent-old"
        assert options.model.id == "gpt-6"
        assert options.mcp_servers == {"memclaw": _MCP}
        assert options.local.cwd == backend._cwd
        assert result.session_id == "agent-old"
        # No system prompt for a resumed agent; it already has one.
        assert "System" not in agent.sent[0][0]
        assert agent.sent[0][1].local.force is True
        assert agent.sent[1][1].local is None

    @pytest.mark.asyncio
    async def test_resume_failure_starts_fresh_agent(self, tmp_path):
        from cursor_sdk import UnknownAgentError

        backend = _backend(tmp_path)
        fresh = _make_agent("agent-new")
        client = _make_client(
            resume=[UnknownAgentError("Agent agent-gone not found")], create=[fresh],
        )

        with _patched(backend, client):
            result = await _turn(backend, resume="agent-gone")

        assert result.session_id == "agent-new"
        assert fresh.sent[0][0].startswith("System")

    @pytest.mark.asyncio
    async def test_failed_turn_drops_agent_and_resumes_next_time(self, tmp_path):
        from cursor_sdk import CursorSDKError

        backend = _backend(tmp_path)
        broken = _make_agent("agent-1")
        broken.send = AsyncMock(side_effect=CursorSDKError("boom"))
        again = _make_agent("agent-1")
        client = _make_client(create=[broken], resume=[again])

        with _patched(backend, client) as launch:
            with pytest.raises(RuntimeError, match="Cursor SDK error: boom"):
                await _turn(backend)
            broken.close.assert_awaited_once()
            result = await _turn(backend, resume="agent-1")

        assert result.session_id == "agent-1"
        launch.assert_awaited_once()  # the bridge itself was fine
        client.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dead_bridge_is_relaunched(self, tmp_path):
        import httpx
        from cursor_sdk import NetworkError

        backend = _backend(tmp_path)
        dead = _make_agent("agent-1")
        err = NetworkError("connection refused")
        err.__cause__ = httpx.ConnectError("refused")
        dead.send = AsyncMock(side_effect=err)
        old_client = _make_client(create=[dead])
        new_client = _make_client(resume=[_make_agent("agent-1")])

        with _patched(backend, old_client, new_client) as launch:
            with pytest.raises(RuntimeError):
                await _turn(backend)
            await _turn(backend, resume="agent-1")

        assert launch.await_count == 2
        old_client.aclose.assert_awaited_once()
        new_client.agents.resume.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reset_session_closes_agent(self, tmp_path):
        backend = _backend(tmp_path)
        first, second = _make_agent("agent-1"), _make_agent("agent-2")
        client = _make_client(create=[first, second])

        with _patched(backend, client):
            await _turn(backend)
            await backend.reset_session("cli")
            result = await _turn(backend)  # agent.py passes no resume id after /new

        first.close.assert_awaited_once()
        assert result.session_id == "agent-2"
        assert second.sent[0][0].startswith("System")

    @pytest.mark.asyncio
    async def test_reset_unknown_session_is_noop(self, tmp_path):
        await _backend(tmp_path).reset_session("nope")

    @pytest.mark.asyncio
    async def test_shutdown_closes_agents_and_bridge(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent()
        client = _make_client(create=[agent])

        with _patched(backend, client), patch.object(
            backend._mcp_server, "stop", new_callable=AsyncMock
        ) as stop_mcp:
            await _turn(backend)
            await backend.on_agent_shutdown()

        agent.close.assert_awaited_once()
        client.aclose.assert_awaited_once()
        stop_mcp.assert_awaited_once()
        assert backend._client is None

    @pytest.mark.asyncio
    async def test_run_turn_applies_cost_fallback(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent()

        async def _send(message, options):
            options.on_delta(
                SimpleNamespace(
                    type="turn-ended",
                    usage={
                        "inputTokens": 1000,
                        "outputTokens": 500,
                        "cacheReadTokens": 200,
                        "cacheWriteTokens": 50,
                    },
                )
            )
            return _make_run()

        agent.send = AsyncMock(side_effect=_send)
        client = _make_client(create=[agent])

        with _patched(backend, client):
            result = await _turn(backend)

        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.cache_read_tokens == 200
        assert result.cache_creation_tokens == 50
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_run_turn_without_mcp_server_raises(self, tmp_path):
        backend = _backend(tmp_path)

        with patch.object(backend, "_ensure_mcp_server", new_callable=AsyncMock), patch.object(
            HttpMcpServer, "config", new_callable=PropertyMock, return_value=None
        ):
            with pytest.raises(RuntimeError, match="MCP server failed to start"):
                await _turn(backend)

    @pytest.mark.asyncio
    async def test_agent_start_starts_mcp_server(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_BACKEND", "cursor")
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        cfg.agent_backend = "cursor"

        from memclaw.agent import MemclawAgent

        agent = MemclawAgent(cfg, platform="telegram")
        assert isinstance(agent.backend, CursorAgentBackend)

        with patch.object(agent.index, "sync", new_callable=AsyncMock), patch.object(
            CursorAgentBackend, "on_agent_start", new_callable=AsyncMock
        ) as mock_start:
            await agent.start()
            mock_start.assert_awaited_once_with(agent._tools)

        with patch.object(
            CursorAgentBackend, "on_agent_shutdown", new_callable=AsyncMock
        ) as mock_stop:
            await agent.aclose()
            mock_stop.assert_awaited_once()


class TestCursorModelSelection:
    @pytest.fixture(autouse=True)
    def _env_file(self, tmp_path, monkeypatch):
        self.env_file = tmp_path / ".env"
        monkeypatch.setattr("memclaw.setup.ENV_FILE", self.env_file)

    def _env(self) -> str:
        return self.env_file.read_text() if self.env_file.exists() else ""

    def test_supports_model_selection(self, tmp_path):
        backend = _backend(tmp_path)
        assert backend.supports_model_selection is True
        assert backend.model == "composer-2.5"
        assert backend.effort is None

    @pytest.mark.asyncio
    async def test_list_models_maps_sdk_models(self, tmp_path):
        backend = _backend(tmp_path)
        client = _make_client()
        client.models.list.return_value = [
            _sdk_model("composer-2.5"),
            _sdk_model("gpt-6", effort_values=("low", "medium", "high")),
        ]

        with _patched(backend, client):
            models = await backend.list_models()

        client.models.list.assert_awaited_once_with(api_key="crsr_test_key")
        assert [(m.id, m.display_name, m.effort_levels) for m in models] == [
            ("composer-2.5", "Composer-2.5", []),
            ("gpt-6", "Gpt-6", ["low", "medium", "high"]),
        ]

    @pytest.mark.asyncio
    async def test_set_model_persists_and_applies_on_next_send(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent(replies=["One", "Two"])
        client = _make_client(create=[agent])
        client.models.list.return_value = [_sdk_model("composer-2.5"), _sdk_model("gpt-6")]

        with _patched(backend, client):
            await _turn(backend)
            assert await backend.set_model("gpt-6") == ""
            await _turn(backend)

        assert backend.model == "gpt-6"
        assert "CURSOR_MODEL=gpt-6" in self._env()
        assert agent.sent[0][1].model.id == "composer-2.5"
        assert agent.sent[1][1].model.id == "gpt-6"
        client.agents.create.assert_awaited_once()  # same agent, no reconnect

    @pytest.mark.asyncio
    async def test_set_unknown_model_raises(self, tmp_path):
        backend = _backend(tmp_path)
        client = _make_client()
        client.models.list.return_value = [_sdk_model("composer-2.5")]

        with _patched(backend, client), pytest.raises(ValueError, match="Unknown model"):
            await backend.set_model("nope")

    @pytest.mark.asyncio
    async def test_effort_is_sent_as_model_parameter(self, tmp_path):
        backend = _backend(tmp_path)
        agent = _make_agent()
        client = _make_client(create=[agent])
        client.models.list.return_value = [
            _sdk_model("gpt-6", effort_values=("low", "high")),
        ]

        with _patched(backend, client):
            await backend.set_model("gpt-6")
            assert await backend.effort_levels() == ["low", "high"]
            await backend.set_effort("HIGH")
            await _turn(backend)

        assert backend.effort == "high"
        assert "CURSOR_EFFORT=high" in self._env()
        selection = agent.sent[0][1].model
        assert selection.id == "gpt-6"
        assert [(p.id, p.value) for p in selection.params] == [("reasoning", "high")]

    @pytest.mark.asyncio
    async def test_effort_rejected_when_model_has_none(self, tmp_path):
        backend = _backend(tmp_path)
        client = _make_client()
        client.models.list.return_value = [_sdk_model("composer-2.5")]

        with _patched(backend, client):
            assert await backend.effort_levels() == []
            with pytest.raises(ValueError, match="doesn't support an effort"):
                await backend.set_effort("high")

    @pytest.mark.asyncio
    async def test_switching_to_model_without_effort_clears_it(self, tmp_path):
        backend = _backend(tmp_path, cursor_effort="high")
        client = _make_client()
        client.models.list.return_value = [
            _sdk_model("gpt-6", effort_values=("low", "high")),
            _sdk_model("composer-2.5"),
        ]

        with _patched(backend, client):
            note = await backend.set_model("composer-2.5")

        assert "doesn't support 'high'" in note
        assert backend.effort is None
        assert "CURSOR_EFFORT" not in self._env()

    @pytest.mark.asyncio
    async def test_model_and_effort_menus_list_cursor_models(self, tmp_path):
        from memclaw.bot.handlers import effort_menu, model_menu, select_model

        backend = _backend(tmp_path)
        client = _make_client()
        client.models.list.return_value = [_sdk_model("composer-2.5"), _sdk_model("gpt-6")]

        with _patched(backend, client):
            _text, keyboard = await model_menu(backend)
            switched = await select_model(backend, "gpt-6")
            effort, effort_keyboard = await effort_menu(backend)

        buttons = [(b.text, b.callback_data) for row in keyboard.inline_keyboard for b in row]
        assert buttons == [("✓ Composer-2.5", "m:composer-2.5"), ("Gpt-6", "m:gpt-6")]
        assert switched.startswith("Switched to gpt-6.")
        assert "doesn't support an effort setting" in effort
        assert effort_keyboard is None


async def _empty_events():
    if False:  # pragma: no cover - async generator helper
        yield


async def _empty_messages():
    if False:  # pragma: no cover - async generator helper
        yield
