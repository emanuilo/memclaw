"""Tests for the Cursor SDK backend."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from cursor_sdk import HttpMcpServerConfig

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
    for name in ("CURSOR_API_KEY", "CURSOR_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _make_config(
    tmp_path: Path,
    *,
    cursor_api_key: str = "",
    cursor_model: str = "",
) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        cursor_api_key=cursor_api_key,
        cursor_model=cursor_model,
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
                "cacheReadInputTokens": 10,
                "cache_creation_input_tokens": 5,
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


class TestCursorAgentBackendRuns:
    @pytest.mark.asyncio
    async def test_run_one_shot(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorAgentBackend(cfg)

        mock_result = SimpleNamespace(result="Hello from Cursor")

        with patch("cursor_sdk.AsyncClient") as mock_client_cls, patch(
            "cursor_sdk.AsyncAgent.prompt", new_callable=AsyncMock
        ) as mock_prompt:
            mock_client = AsyncMock()
            mock_client_cls.launch_bridge = AsyncMock(return_value=mock_client)
            mock_prompt.return_value = mock_result

            text = await backend.run_one_shot(
                system_prompt="System",
                user_message="User",
            )

        assert text == "Hello from Cursor"
        mock_prompt.assert_awaited_once()
        call_prompt = mock_prompt.await_args.args[0]
        assert "System" in call_prompt
        assert "User" in call_prompt

    @pytest.mark.asyncio
    async def test_run_one_shot_missing_key_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        backend = CursorAgentBackend(cfg)

        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            await backend.run_one_shot(system_prompt="S", user_message="U")

    @pytest.mark.asyncio
    async def test_run_turn(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorAgentBackend(cfg)

        mock_run = AsyncMock()
        mock_run.events = MagicMock(return_value=_empty_events())
        mock_run.wait = AsyncMock(return_value=SimpleNamespace(result="Turn response", num_turns=2))
        mock_agent = AsyncMock()
        mock_agent.send = AsyncMock(return_value=mock_run)
        mock_agent.close = AsyncMock()
        mock_client = AsyncMock()
        mock_client.agents.create = AsyncMock(return_value=mock_agent)
        mock_mcp = HttpMcpServerConfig(url="http://127.0.0.1:8765/mcp", type="http")

        with patch("cursor_sdk.AsyncClient") as mock_client_cls, patch.object(
            backend, "_ensure_mcp_server", new_callable=AsyncMock
        ), patch.object(
            HttpMcpServer, "config", new_callable=PropertyMock, return_value=mock_mcp
        ):
            mock_client_cls.launch_bridge = AsyncMock(return_value=mock_client)

            result = await backend.run_turn(
                system_prompt="System",
                user_message="User",
                tool_executor=MagicMock(),
                max_turns=5,
            )

        assert result.text == "Turn response"
        assert result.num_turns == 2
        assert result.cost_usd is None
        send_options = mock_agent.send.await_args.args[1]
        assert send_options.on_delta is not None
        mock_client.agents.create.assert_awaited_once()
        create_options = mock_client.agents.create.await_args.args[0]
        assert create_options.mcp_servers == {"memclaw": mock_mcp}
        send_options = mock_agent.send.await_args.args[1]
        assert send_options.mcp_servers == {"memclaw": mock_mcp}
        mock_agent.send.assert_awaited_once()
        mock_agent.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_turn_applies_cost_fallback(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorAgentBackend(cfg)

        async def _events():
            yield SimpleNamespace(sdk_message=None, interaction_update=None, result=None)

        mock_run = AsyncMock()
        mock_run.events = MagicMock(return_value=_events())
        mock_run.wait = AsyncMock(return_value=SimpleNamespace(result="Turn response", num_turns=1))

        captured: dict[str, object] = {}

        async def _send(message, options):
            options.on_delta(
                SimpleNamespace(
                    type="turn-ended",
                    usage={"input_tokens": 1000, "output_tokens": 500},
                )
            )
            return mock_run

        mock_agent = AsyncMock()
        mock_agent.send = AsyncMock(side_effect=_send)
        mock_agent.close = AsyncMock()
        mock_client = AsyncMock()
        mock_client.agents.create = AsyncMock(return_value=mock_agent)
        mock_mcp = HttpMcpServerConfig(url="http://127.0.0.1:8765/mcp", type="http")

        with patch("cursor_sdk.AsyncClient") as mock_client_cls, patch.object(
            backend, "_ensure_mcp_server", new_callable=AsyncMock
        ), patch.object(
            HttpMcpServer, "config", new_callable=PropertyMock, return_value=mock_mcp
        ):
            mock_client_cls.launch_bridge = AsyncMock(return_value=mock_client)

            result = await backend.run_turn(
                system_prompt="System",
                user_message="User",
                tool_executor=MagicMock(),
                max_turns=5,
            )

        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_run_turn_without_mcp_server_raises(self, tmp_path):
        cfg = _make_config(tmp_path, cursor_api_key="crsr_test_key")
        backend = CursorAgentBackend(cfg)

        with patch.object(backend, "_ensure_mcp_server", new_callable=AsyncMock), patch.object(
            HttpMcpServer, "config", new_callable=PropertyMock, return_value=None
        ):
            with pytest.raises(RuntimeError, match="MCP server failed to start"):
                await backend.run_turn(
                    system_prompt="System",
                    user_message="User",
                    tool_executor=MagicMock(),
                )

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


async def _empty_events():
    if False:  # pragma: no cover - async generator helper
        yield


async def _empty_messages():
    if False:  # pragma: no cover - async generator helper
        yield
