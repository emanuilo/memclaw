"""Tests for the Claude Agent SDK backend."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memclaw.backends import claude as claude_backend
from memclaw.backends.claude import ClaudeAgentBackend
from memclaw.config import MemclawConfig


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Prevent the developer's shell env from leaking into `MemclawConfig`.

    MemclawConfig.__post_init__ falls back to os.environ when fields are
    blank, so a real CLAUDE_CODE_OAUTH_TOKEN in the parent shell would
    silently override `_make_config(api_key=...)`.
    """
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
                 "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _make_config(tmp_path: Path, *, oauth: str = "", api_key: str = "") -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="test-openai-key",
        anthropic_api_key=api_key,
        claude_code_oauth_token=oauth,
    )


def _mock_sdk_client(text: str):
    """Build a fake ClaudeSDKClient context that yields one AssistantMessage
    followed by a ResultMessage. Returns (ctx_factory, client_mock).
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    assistant = AssistantMessage(
        content=[TextBlock(text=text)],
        model="claude-sonnet-4-6",
    )
    result = ResultMessage(
        subtype="result",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="test",
        total_cost_usd=None,
        usage={"input_tokens": 100, "output_tokens": 50,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        result=None,
    )

    client = MagicMock()
    client.query = AsyncMock()

    def _receive_factory():
        async def _gen():
            yield assistant
            yield result
        return _gen()

    client.receive_response = MagicMock(side_effect=_receive_factory)

    def _ctx_factory(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    return _ctx_factory, client


# ────────────────────────────────────────────────────────────────────
# Auth mode + env scrubbing
# ────────────────────────────────────────────────────────────────────

class TestAuthMode:
    def test_subscription_when_oauth_set(self, tmp_path: Path):
        cfg = _make_config(tmp_path, oauth="oauth-token")
        assert claude_backend._claude_auth_mode(cfg) == "subscription"

    def test_api_key_when_only_api_key_set(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="sk-ant-test")
        assert claude_backend._claude_auth_mode(cfg) == "api_key"

    def test_oauth_wins_when_both_set(self, tmp_path: Path):
        cfg = _make_config(tmp_path, oauth="oauth-token", api_key="sk-ant-test")
        assert claude_backend._claude_auth_mode(cfg) == "subscription"

    def test_empty_when_neither_set(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        assert claude_backend._claude_auth_mode(cfg) == ""

    def test_bills_per_token_only_for_api_key(self, tmp_path: Path):
        sub = ClaudeAgentBackend(_make_config(tmp_path, oauth="oauth-token"))
        api = ClaudeAgentBackend(_make_config(tmp_path, api_key="sk-ant-test"))
        assert sub.bills_per_token is False
        assert api.bills_per_token is True


class TestBuildEnv:
    def test_strips_stale_credentials(self, tmp_path: Path):
        cfg = _make_config(tmp_path, oauth="my-oauth")
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "stale-key",
            "ANTHROPIC_AUTH_TOKEN": "stale-token",
            "CLAUDE_CODE_OAUTH_TOKEN": "stale-oauth",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "PATH": "/usr/bin",
        }, clear=True):
            env = claude_backend._build_env(cfg)

        # Stale credentials are dropped; the chosen one is set.
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "CLAUDE_CODE_USE_BEDROCK" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "my-oauth"
        # Unrelated env survives.
        assert env["PATH"] == "/usr/bin"

    def test_injects_api_key_when_configured(self, tmp_path: Path):
        cfg = _make_config(tmp_path, api_key="my-api-key")
        with patch.dict(os.environ, {}, clear=True):
            env = claude_backend._build_env(cfg)
        assert env["ANTHROPIC_API_KEY"] == "my-api-key"
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    def test_no_credential_injected_when_unconfigured(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        with patch.dict(os.environ, {}, clear=True):
            env = claude_backend._build_env(cfg)
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


# ────────────────────────────────────────────────────────────────────
# is_configured
# ────────────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_oauth_token_satisfies(self, tmp_path: Path):
        assert ClaudeAgentBackend.is_configured(_make_config(tmp_path, oauth="x"))

    def test_api_key_satisfies(self, tmp_path: Path):
        assert ClaudeAgentBackend.is_configured(_make_config(tmp_path, api_key="x"))

    def test_neither_fails(self, tmp_path: Path):
        assert not ClaudeAgentBackend.is_configured(_make_config(tmp_path))


# ────────────────────────────────────────────────────────────────────
# Runtime: run_one_shot + run_turn
# ────────────────────────────────────────────────────────────────────

class TestRunOneShot:
    @pytest.mark.asyncio
    async def test_returns_text(self, tmp_path: Path):
        backend = ClaudeAgentBackend(_make_config(tmp_path, oauth="x"))
        ctx_factory, client = _mock_sdk_client("hello world")
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=ctx_factory):
            text = await backend.run_one_shot(
                system_prompt="be nice", user_message="hi",
            )
        assert text == "hello world"
        # The user message reached the SDK exactly once.
        assert client.query.await_count == 1
        assert client.query.await_args.args[0] == "hi"


class TestRunTurn:
    @pytest.mark.asyncio
    async def test_returns_turn_result(self, tmp_path: Path):
        from memclaw.tools import ToolExecutor

        backend = ClaudeAgentBackend(_make_config(tmp_path, api_key="x"))
        cfg = backend.config
        executor = ToolExecutor(
            config=cfg,
            store=MagicMock(),
            index=MagicMock(),
            search=MagicMock(),
            found_images=[],
            platform="test",
        )

        ctx_factory, _client = _mock_sdk_client("done")
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=ctx_factory):
            result = await backend.run_turn(
                system_prompt="sys",
                user_message="hello",
                tool_executor=executor,
            )

        assert result.text == "done"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        # bills_per_token is True (api_key) and the mock returns no cost,
        # so the backend should compute the fallback cost.
        assert result.cost_usd is not None
        assert result.cost_usd > 0
