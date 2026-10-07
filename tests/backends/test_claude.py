"""Tests for the Claude Agent SDK backend."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memclaw.backends import claude as claude_backend
from memclaw.backends.claude import ClaudeAgentBackend
from memclaw.backends.claude_models import ModelInfo


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

# Every test here runs with the developer's Claude env vars cleared.
pytestmark = pytest.mark.usefixtures("isolate_claude_env")


def _mock_sdk_client(text: str, *, session_id: str = "test", costs=None):
    """Build a fake ClaudeSDKClient context that yields one AssistantMessage
    followed by a ResultMessage. Returns (ctx_factory, client_mock).
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    assistant = AssistantMessage(
        content=[TextBlock(text=text)],
        model="claude-sonnet-5-5",
    )
    result = ResultMessage(
        subtype="result",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        total_cost_usd=None,
        usage={"input_tokens": 100, "output_tokens": 50,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        result=None,
    )

    client = MagicMock()
    client.query = AsyncMock()

    # *costs*: the running session total the CLI reports, one per query.
    cost_iter = iter(costs or ())

    def _receive_factory():
        async def _gen():
            yield assistant
            result.total_cost_usd = next(cost_iter, None)
            yield result
        return _gen()

    client.receive_response = MagicMock(side_effect=_receive_factory)

    def _ctx_factory(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    return _ctx_factory, client


def _model_info(model_id: str, *, effort_levels: list[str] | None = None):
    return ModelInfo(
        id=model_id,
        display_name=model_id,
        created_at="2026-07-24T00:00:00Z",
        effort_levels=effort_levels or [],
    )


# ────────────────────────────────────────────────────────────────────
# Auth mode + env scrubbing
# ────────────────────────────────────────────────────────────────────

class TestAuthMode:
    def test_subscription_when_oauth_set(self, claude_config):
        cfg = claude_config(oauth="oauth-token")
        assert claude_backend._claude_auth_mode(cfg) == "subscription"

    def test_api_key_when_only_api_key_set(self, claude_config):
        cfg = claude_config(api_key="sk-ant-test")
        assert claude_backend._claude_auth_mode(cfg) == "api_key"

    def test_oauth_wins_when_both_set(self, claude_config):
        cfg = claude_config(oauth="oauth-token", api_key="sk-ant-test")
        assert claude_backend._claude_auth_mode(cfg) == "subscription"

    def test_empty_when_neither_set(self, claude_config):
        cfg = claude_config()
        assert claude_backend._claude_auth_mode(cfg) == ""

    def test_bills_per_token_only_for_api_key(self, claude_config):
        sub = ClaudeAgentBackend(claude_config(oauth="oauth-token"))
        api = ClaudeAgentBackend(claude_config(api_key="sk-ant-test"))
        assert sub.bills_per_token is False
        assert api.bills_per_token is True


class TestBuildEnv:
    def test_strips_stale_credentials(self, claude_config):
        cfg = claude_config(oauth="my-oauth")
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

    def test_injects_api_key_when_configured(self, claude_config):
        cfg = claude_config(api_key="my-api-key")
        with patch.dict(os.environ, {}, clear=True):
            env = claude_backend._build_env(cfg)
        assert env["ANTHROPIC_API_KEY"] == "my-api-key"
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    def test_no_credential_injected_when_unconfigured(self, claude_config):
        cfg = claude_config()
        with patch.dict(os.environ, {}, clear=True):
            env = claude_backend._build_env(cfg)
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


# ────────────────────────────────────────────────────────────────────
# is_configured
# ────────────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_oauth_token_satisfies(self, claude_config):
        assert ClaudeAgentBackend.is_configured(claude_config(oauth="x"))

    def test_api_key_satisfies(self, claude_config):
        assert ClaudeAgentBackend.is_configured(claude_config(api_key="x"))

    def test_neither_fails(self, claude_config):
        assert not ClaudeAgentBackend.is_configured(claude_config())


# ────────────────────────────────────────────────────────────────────
# Runtime: run_one_shot + run_turn
# ────────────────────────────────────────────────────────────────────

class TestRunOneShot:
    @pytest.mark.asyncio
    async def test_returns_text(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        ctx_factory, client = _mock_sdk_client("hello world")
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=ctx_factory):
            text = await backend.run_one_shot(
                system_prompt="be nice", user_message="hi",
            )
        assert text == "hello world"
        # The user message reached the SDK exactly once.
        assert client.query.await_count == 1
        assert client.query.await_args.args[0] == "hi"


def _executor(cfg):
    from memclaw.tools import ToolExecutor

    return ToolExecutor(
        config=cfg, store=MagicMock(), index=MagicMock(),
        search=MagicMock(), found_images=[], platform="test",
    )


class _FakeClients:
    """Stands in for ClaudeSDKClient: records every client built (with its
    options) and serves one canned reply per query."""

    def __init__(self, text: str = "done", *, session_id: str = "sess-1",
                 fail_resume: bool = False, costs=None):
        self.costs = costs
        self.text = text
        self.session_id = session_id
        self.fail_resume = fail_resume
        self.clients: list[MagicMock] = []
        self.options: list = []

    def __call__(self, options=None, *args, **kwargs):
        _factory, client = _mock_sdk_client(
            self.text, session_id=self.session_id, costs=self.costs,
        )
        resume = options.resume if options is not None else None
        client.connect = AsyncMock(
            side_effect=RuntimeError("no such session") if self.fail_resume and resume else None,
        )
        client.disconnect = AsyncMock()
        self.clients.append(client)
        self.options.append(options)
        return client


async def _turn(backend, executor, *, key="telegram:1", resume=None,
                context="<context>now</context>", message="hello", **kwargs):
    return await backend.run_turn(
        system_prompt="sys", context=context, user_message=message,
        tool_executor=executor, session_key=key, resume_session_id=resume,
        **kwargs,
    )


class TestRunTurn:
    @pytest.mark.asyncio
    async def test_returns_turn_result(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(api_key="x"))
        fake = _FakeClients("done", session_id="sess-1")
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            result = await _turn(backend, _executor(backend.config))

        assert result.text == "done"
        assert result.session_id == "sess-1"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        # bills_per_token is True (api_key) and the mock returns no cost,
        # so the backend should compute the fallback cost.
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_fallback_cost_uses_the_configured_models_price(self, claude_config):
        cfg = claude_config(api_key="x")
        cfg.claude_model = "claude-opus-5"
        backend = ClaudeAgentBackend(cfg)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=_FakeClients()):
            result = await _turn(backend, _executor(cfg))

        # 100 input tokens at $5/M + 50 output tokens at $25/M (Opus 5).
        assert result.cost_usd == pytest.approx((100 * 5 + 50 * 25) / 1_000_000)

    @pytest.mark.asyncio
    async def test_cost_is_per_turn_not_the_session_running_total(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(api_key="x"))
        fake = _FakeClients(costs=[0.01, 0.025])
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            first = await _turn(backend, executor)
            second = await _turn(backend, executor, resume="sess-1")

        assert first.cost_usd == pytest.approx(0.01)
        assert second.cost_usd == pytest.approx(0.015)

    @pytest.mark.asyncio
    async def test_resumed_session_without_baseline_estimates_cost(self, claude_config):
        # After a restart the running total includes earlier turns, so the
        # token-based estimate is used instead.
        backend = ClaudeAgentBackend(claude_config(api_key="x"))
        fake = _FakeClients(costs=[5.0])
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            result = await _turn(backend, _executor(backend.config), resume="sess-1")

        input_per_m, output_per_m = claude_backend._prices_per_m(backend.model)
        assert result.cost_usd == pytest.approx((100 * input_per_m + 50 * output_per_m) / 1_000_000)

    @pytest.mark.asyncio
    async def test_session_options_are_locked_down(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, _executor(backend.config), max_turns=7)

        options = fake.options[0]
        assert options.setting_sources == []
        assert options.strict_mcp_config is True
        assert options.tools == []
        assert options.permission_mode == "dontAsk"
        assert options.verbatim_prompts is True
        assert options.allowed_tools == claude_backend._ALLOWED_TOOLS
        assert all(t.startswith("mcp__memclaw__") for t in options.allowed_tools)
        assert options.disallowed_tools == []
        assert options.system_prompt == "sys"
        assert options.cwd == str(backend.config.memory_dir)
        assert options.max_turns == 7
        assert options.resume is None

    @pytest.mark.asyncio
    async def test_context_block_is_prefixed_to_the_message(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, _executor(backend.config),
                        context="<context>ctx</context>", message="hi there")

        prompt = fake.clients[0].query.await_args.args[0]
        assert prompt == "<context>ctx</context>\n\nhi there"

    @pytest.mark.asyncio
    async def test_image_turn_streams_context_in_the_text_part(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, _executor(backend.config),
                        context="<context>ctx</context>", message="a photo",
                        image_b64="aGVsbG8=", image_media_type="image/png")

        stream = fake.clients[0].query.await_args.args[0]
        messages = [m async for m in stream]
        content = messages[0]["message"]["content"]
        assert content[0]["source"] == {
            "type": "base64", "media_type": "image/png", "data": "aGVsbG8=",
        }
        assert content[1] == {"type": "text", "text": "<context>ctx</context>\n\na photo"}


class TestPersistentSessions:
    @pytest.mark.asyncio
    async def test_client_is_reused_across_turns(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor, message="one")
            await _turn(backend, executor, message="two", resume="sess-1")

        assert len(fake.clients) == 1
        client = fake.clients[0]
        client.connect.assert_awaited_once()
        assert client.query.await_count == 2
        client.disconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_each_chat_gets_its_own_client(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor, key="telegram:1")
            await _turn(backend, executor, key="telegram:2")
        assert len(fake.clients) == 2

    @pytest.mark.asyncio
    async def test_resume_id_reaches_the_cli(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, _executor(backend.config), resume="old-session")
        assert fake.options[0].resume == "old-session"

    @pytest.mark.asyncio
    async def test_failed_resume_falls_back_to_a_fresh_session(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients("fresh", session_id="new-session", fail_resume=True)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            result = await _turn(backend, _executor(backend.config), resume="gone")

        assert [o.resume for o in fake.options] == ["gone", None]
        assert result.text == "fresh"
        assert result.session_id == "new-session"

    @pytest.mark.asyncio
    async def test_reset_session_closes_the_client(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor)
            await backend.reset_session("telegram:1")
            fake.clients[0].disconnect.assert_awaited_once()
            await _turn(backend, executor)

        assert len(fake.clients) == 2
        assert fake.options[1].resume is None

    @pytest.mark.asyncio
    async def test_reset_of_an_unknown_session_is_a_no_op(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        await backend.reset_session("nope")

    @pytest.mark.asyncio
    async def test_failed_turn_drops_the_client_so_the_next_one_resumes(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor)
            fake.clients[0].query = AsyncMock(side_effect=RuntimeError("CLI died"))
            with pytest.raises(RuntimeError):
                await _turn(backend, executor, resume="sess-1")
            fake.clients[0].disconnect.assert_awaited_once()
            await _turn(backend, executor, resume="sess-1")

        assert len(fake.clients) == 2
        assert fake.options[1].resume == "sess-1"

    @pytest.mark.asyncio
    async def test_model_change_reconnects_resuming_the_session(
        self, claude_config, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr("memclaw.setup.ENV_FILE", tmp_path / ".env")
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        backend._models = [_model_info("claude-opus-5", effort_levels=["low", "medium"])]
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor)
            await backend.set_model("claude-opus-5")
            # The live client is only replaced on the next turn.
            fake.clients[0].disconnect.assert_not_awaited()
            await _turn(backend, executor, resume="sess-1")

        fake.clients[0].disconnect.assert_awaited_once()
        assert len(fake.clients) == 2
        assert fake.options[1].resume == "sess-1"
        assert fake.options[1].model == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_shutdown_closes_every_client(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))
        fake = _FakeClients()
        executor = _executor(backend.config)
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=fake):
            await _turn(backend, executor, key="a")
            await _turn(backend, executor, key="b")
        await backend.on_agent_shutdown()
        for client in fake.clients:
            client.disconnect.assert_awaited_once()


class TestModelSelection:
    @pytest.fixture
    def env_file(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("OPENAI_API_KEY=k\nCLAUDE_EFFORT=high\n")
        monkeypatch.setattr("memclaw.setup.ENV_FILE", path)
        return path

    MODELS = [
        _model_info("claude-opus-5", effort_levels=["low", "medium", "high", "max"]),
        _model_info("claude-sonnet-5-5", effort_levels=["low", "medium"]),
        _model_info("claude-haiku-4-5"),
    ]

    def _backend(self, claude_config, effort="high"):
        cfg = claude_config(oauth="x")
        cfg.claude_model = "claude-opus-5"
        cfg.claude_effort = effort
        backend = ClaudeAgentBackend(cfg)
        return backend

    @pytest.mark.asyncio
    async def test_list_models_uses_the_configured_credential(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="tok"))
        fetch = AsyncMock(return_value=self.MODELS)
        with patch("memclaw.backends.claude.fetch_models", fetch):
            assert await backend.list_models() == self.MODELS
        fetch.assert_awaited_once_with("subscription", "tok")

    @pytest.mark.asyncio
    async def test_set_model_persists_and_updates_config(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        note = await backend.set_model("claude-opus-5")
        assert note == ""
        assert backend.model == "claude-opus-5"
        assert backend.effort == "high"
        assert backend.config.claude_model == "claude-opus-5"
        text = env_file.read_text()
        assert "CLAUDE_MODEL=claude-opus-5" in text
        assert "CLAUDE_EFFORT=high" in text
        assert "OPENAI_API_KEY=k" in text

    @pytest.mark.asyncio
    async def test_unsupported_effort_steps_down(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        note = await backend.set_model("claude-sonnet-5-5")
        assert backend.effort == "medium"
        assert "'high'" in note and "'medium'" in note
        assert "CLAUDE_EFFORT=medium" in env_file.read_text()

    @pytest.mark.asyncio
    async def test_model_without_effort_drops_it(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        note = await backend.set_model("claude-haiku-4-5")
        assert backend.effort is None
        assert "effort" in note
        text = env_file.read_text()
        assert "CLAUDE_MODEL=claude-haiku-4-5" in text
        assert "CLAUDE_EFFORT" not in text

    @pytest.mark.asyncio
    async def test_unknown_model_is_rejected(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        with pytest.raises(ValueError, match="Unknown model"):
            await backend.set_model("claude-nope")
        assert backend.model == "claude-opus-5"

    @pytest.mark.asyncio
    async def test_any_id_is_accepted_when_the_list_is_unavailable(
        self, claude_config, env_file,
    ):
        backend = self._backend(claude_config)
        with patch("memclaw.backends.claude.fetch_models",
                   AsyncMock(side_effect=httpx.ConnectError("offline"))):
            await backend.set_model("claude-future-9")
        assert backend.model == "claude-future-9"
        assert backend.effort == "high"

    @pytest.mark.asyncio
    async def test_set_effort(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        await backend.set_effort(" MAX ")
        assert backend.effort == "max"
        assert "CLAUDE_EFFORT=max" in env_file.read_text()

    @pytest.mark.asyncio
    async def test_set_effort_rejects_unsupported_levels(self, claude_config, env_file):
        backend = self._backend(claude_config)
        backend._models = list(self.MODELS)
        with pytest.raises(ValueError, match="Unknown effort"):
            await backend.set_effort("ultra")
        with pytest.raises(ValueError, match="doesn't support 'xhigh'"):
            await backend.set_effort("xhigh")
        assert backend.effort == "high"

    @pytest.mark.asyncio
    async def test_set_effort_on_a_model_without_effort(self, claude_config, env_file):
        backend = self._backend(claude_config, effort="")
        backend._model = "claude-haiku-4-5"
        backend._models = list(self.MODELS)
        assert await backend.effort_levels() == []
        with pytest.raises(ValueError, match="doesn't support an effort"):
            await backend.set_effort("low")


class TestPrices:
    @pytest.mark.parametrize("model, prices", [
        ("claude-opus-5", (5.0, 25.0)),
        ("claude-opus-5-5", (4.0, 20.0)),         # not mistaken for claude-opus-5
        ("claude-sonnet-4-6", (3.0, 15.0)),
        ("claude-sonnet-5-5", (2.0, 10.0)),
        ("claude-haiku-4-5-20251001", (1.0, 5.0)),  # dated snapshot
    ])
    def test_known_models(self, model, prices):
        assert claude_backend._prices_per_m(model) == prices

    @pytest.mark.parametrize("model", ["opus", "claude-future-9"])
    def test_unknown_model_uses_the_default_models_price(self, model):
        assert claude_backend._prices_per_m(model) == claude_backend._prices_per_m(
            claude_backend._DEFAULT_MODEL,
        )


# ────────────────────────────────────────────────────────────────────
# Model + effort resolution
# ────────────────────────────────────────────────────────────────────

class TestResolveModel:
    def test_falls_back_to_default_when_unset(self, claude_config):
        cfg = claude_config(oauth="x")
        assert claude_backend._resolve_model(cfg) == claude_backend._DEFAULT_MODEL

    def test_configured_model_wins(self, claude_config):
        cfg = claude_config(oauth="x")
        cfg.claude_model = "claude-opus-5"
        assert claude_backend._resolve_model(cfg) == "claude-opus-5"

    def test_blank_value_falls_back(self, claude_config):
        cfg = claude_config(oauth="x")
        cfg.claude_model = "   "
        assert claude_backend._resolve_model(cfg) == claude_backend._DEFAULT_MODEL


class TestResolveEffort:
    def test_unset_gives_the_default_on_the_default_model(self, claude_config):
        cfg = claude_config(oauth="x")
        assert claude_backend._resolve_effort(cfg) == claude_backend._DEFAULT_EFFORT == "medium"

    def test_unset_gives_none_on_another_model(self, claude_config):
        """Another model may not take an effort at all, so leave it to the model."""
        cfg = claude_config(oauth="x")
        cfg.claude_model = "claude-haiku-4-5"
        assert claude_backend._resolve_effort(cfg) is None

    def test_known_level_passes_through(self, claude_config):
        cfg = claude_config(oauth="x")
        cfg.claude_effort = "xhigh"
        assert claude_backend._resolve_effort(cfg) == "xhigh"

    def test_case_and_padding_are_normalised(self, claude_config):
        cfg = claude_config(oauth="x")
        cfg.claude_effort = "  HIGH  "
        assert claude_backend._resolve_effort(cfg) == "high"

    def test_level_the_sdk_does_not_know_is_ignored(self, claude_config):
        """A typo in ~/.memclaw/.env must not reach the CLI as an argument."""
        cfg = claude_config(oauth="x")
        cfg.claude_model = "claude-opus-5"
        cfg.claude_effort = "ultra"
        assert claude_backend._resolve_effort(cfg) is None


class TestStatusRows:
    def test_configured_values_are_shown(self, claude_config):
        cfg = claude_config(api_key="k")
        cfg.claude_model = "claude-opus-5"
        cfg.claude_effort = "max"
        assert ClaudeAgentBackend.status_rows(cfg) == [
            ("Model", "claude-opus-5"), ("Effort", "max"),
        ]

    def test_defaults_are_shown_when_unset(self, claude_config):
        rows = ClaudeAgentBackend.status_rows(claude_config(api_key="k"))
        assert rows == [("Model", "claude-sonnet-5-5"), ("Effort", "medium")]


class TestOptionsCarryModelAndEffort:
    @pytest.mark.asyncio
    async def test_configured_values_reach_the_sdk(self, claude_config):
        cfg = claude_config(oauth="x")
        cfg.claude_model = "claude-opus-5"
        cfg.claude_effort = "max"
        backend = ClaudeAgentBackend(cfg)

        ctx_factory, _client = _mock_sdk_client("ok")
        seen = {}

        def _capture(options, *args, **kwargs):
            seen["options"] = options
            return ctx_factory()

        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=_capture):
            await backend.run_one_shot(system_prompt="s", user_message="u")

        assert seen["options"].model == "claude-opus-5"
        assert seen["options"].effort == "max"

    @pytest.mark.asyncio
    async def test_defaults_reach_the_sdk_when_nothing_configured(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))

        ctx_factory, _client = _mock_sdk_client("ok")
        seen = {}

        def _capture(options, *args, **kwargs):
            seen["options"] = options
            return ctx_factory()

        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=_capture):
            await backend.run_one_shot(system_prompt="s", user_message="u")

        assert seen["options"].model == "claude-sonnet-5-5"
        assert seen["options"].effort == "medium"

    @pytest.mark.asyncio
    async def test_one_shot_uses_the_fast_options(self, claude_config):
        backend = ClaudeAgentBackend(claude_config(oauth="x"))

        ctx_factory, _client = _mock_sdk_client("ok")
        seen = {}

        def _capture(options, *args, **kwargs):
            seen["options"] = options
            return ctx_factory()

        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=_capture):
            await backend.run_one_shot(system_prompt="s", user_message="u")

        options = seen["options"]
        assert options.setting_sources == []
        assert options.strict_mcp_config is True
        assert options.tools == []
        assert options.max_turns == 1


# ────────────────────────────────────────────────────────────────────
# Wizard: model + effort questions
# ────────────────────────────────────────────────────────────────────

def _ask(models, *, existing=None, answers=("1",),
         fetch_error: Exception | None = None):
    """Run `_ask_model_and_effort` with the fetch and the prompts mocked.

    Returns (values, drop_keys, prompts, console).
    """
    console = MagicMock()   # MagicMock covers console.status(...) as a context manager

    fetch = AsyncMock(side_effect=fetch_error) if fetch_error else AsyncMock(
        return_value=models)
    prompts: list[dict] = []

    def _prompt(text, **kwargs):
        prompts.append({"text": text, **kwargs})
        return answers[len(prompts) - 1]

    with patch("memclaw.backends.claude.fetch_models", fetch), \
            patch("memclaw.prompts.Prompt.ask", side_effect=_prompt):
        values, drops = ClaudeAgentBackend._ask_model_and_effort(
            console, existing or {}, auth_mode="api_key", credential="sk-ant-test",
        )
    fetch.assert_awaited_once_with("api_key", "sk-ant-test")
    return values, drops, prompts, console


class TestWizardModelQuestion:
    def test_picked_model_and_effort_are_stored(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "high", "max"])]
        values, drops, prompts, _ = _ask(models, answers=("1", "3"))
        assert values == {"CLAUDE_MODEL": "claude-opus-5", "CLAUDE_EFFORT": "max"}
        assert drops == []
        assert len(prompts) == 2

    def test_effort_question_is_skipped_without_support(self):
        """A model reporting no effort level is never asked about one, and any
        level left over from an earlier choice is dropped."""
        models = [_model_info("claude-haiku-4-5")]
        values, drops, prompts, _ = _ask(
            models, existing={"CLAUDE_EFFORT": "high"}, answers=("1",),
        )
        assert values == {"CLAUDE_MODEL": "claude-haiku-4-5"}
        assert drops == ["CLAUDE_EFFORT"]
        assert len(prompts) == 1

    def test_effort_defaults_to_medium(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "medium", "high"])]
        _, _, prompts, _ = _ask(models, answers=("1", "2"))
        assert prompts[1]["default"] == "2"

    def test_effort_default_falls_back_when_medium_is_unsupported(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "high"])]
        _, _, prompts, _ = _ask(models, answers=("1", "1"))
        assert prompts[1]["default"] == "1"

    def test_current_model_is_preselected(self):
        models = [_model_info("claude-opus-5"), _model_info("claude-sonnet-5")]
        _, _, prompts, _ = _ask(
            models,
            existing={"CLAUDE_MODEL": "claude-sonnet-5"}, answers=("2",),
        )
        assert prompts[0]["default"] == "2"

    def test_current_effort_is_preselected(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "medium", "high"])]
        _, _, prompts, _ = _ask(
            models,
            existing={"CLAUDE_EFFORT": "low"}, answers=("1", "1"),
        )
        assert prompts[1]["default"] == "1"


class TestWizardKeepsCurrentModel:
    """Accepting the defaults must never change the model."""

    MODELS = [_model_info("claude-opus-5", effort_levels=["high"]),
              _model_info("claude-sonnet-5")]

    def test_alias_not_in_the_list_stays_the_default(self):
        _, _, prompts, _ = _ask(
            self.MODELS, existing={"CLAUDE_MODEL": "opus"}, answers=("3",),
        )
        assert prompts[0]["choices"] == ["1", "2", "3"]
        assert prompts[0]["default"] == "3"

    def test_keeping_it_changes_nothing(self):
        values, drops, prompts, _ = _ask(
            self.MODELS,
            existing={"CLAUDE_MODEL": "opus", "CLAUDE_EFFORT": "max"},
            answers=("3",),
        )
        assert (values, drops) == ({}, [])
        assert len(prompts) == 1     # no effort question for an unknown model

    def test_a_listed_model_can_still_be_picked(self):
        values, _, _, _ = _ask(
            self.MODELS, existing={"CLAUDE_MODEL": "opus"}, answers=("2",),
        )
        assert values == {"CLAUDE_MODEL": "claude-sonnet-5"}

    def test_model_set_only_in_the_shell_counts_as_current(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")
        _, _, prompts, _ = _ask(self.MODELS, answers=("2",))
        assert prompts[0]["default"] == "2"

    def test_effort_set_only_in_the_shell_counts_as_current(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_EFFORT", "LOW")
        models = [_model_info("claude-opus-5", effort_levels=["low", "high"])]
        _, _, prompts, _ = _ask(models, answers=("1", "1"))
        assert prompts[1]["default"] == "1"

    def test_unset_default_model_missing_from_the_list_is_kept(self):
        """With nothing configured, the built-in default is the current model."""
        values, _, prompts, _ = _ask(self.MODELS, answers=("3",))
        assert prompts[0]["default"] == "3"
        assert values == {}


class TestWizardSurvivesFetchFailure:
    @pytest.mark.parametrize("error", [
        httpx.ConnectError("no network"),
        httpx.ReadTimeout("timed out"),
        RuntimeError("no Claude credential configured"),
    ])
    def test_failure_warns_and_keeps_the_current_value(self, error):
        values, drops, prompts, console = _ask([], fetch_error=error)
        assert values == {}
        assert drops == []
        assert prompts == []          # nothing was asked
        assert _warned(console)

    def test_empty_model_list_warns_and_keeps_the_current_value(self):
        values, drops, prompts, console = _ask([])
        assert (values, drops, prompts) == ({}, [], [])
        assert _warned(console)


def _warned(console) -> bool:
    """True when the console got exactly one yellow warning line."""
    printed = [c.args[0] for c in console.print.call_args_list
               if c.args and isinstance(c.args[0], str)]
    return sum("[yellow]" in line for line in printed) == 1


# ────────────────────────────────────────────────────────────────────
# Wizard: the whole wizard_setup flow
# ────────────────────────────────────────────────────────────────────

def _wizard(*, answers, credential="", existing=None, models=None,
            fetch_error: Exception | None = None):
    """Run `wizard_setup` end to end with every prompt and the fetch mocked.

    *answers* feeds the numbered pickers in order (auth, model, effort);
    *credential* is what the user types at the masked credential prompt.
    Returns (values, drop_keys, fetch_mock).
    """
    fetch = AsyncMock(side_effect=fetch_error) if fetch_error else AsyncMock(
        return_value=models or [])
    answer_iter = iter(answers)

    with patch("memclaw.prompts.Prompt.ask",
               side_effect=lambda *a, **k: next(answer_iter)), \
            patch("memclaw.setup._masked_input", return_value=credential), \
            patch("memclaw.backends.claude.fetch_models", fetch):
        values, drops = ClaudeAgentBackend.wizard_setup(MagicMock(), existing or {})
    return values, drops, fetch


class TestWizardSetupFlow:
    def test_subscription_with_an_effort_model(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "high"])]
        values, drops, fetch = _wizard(
            answers=("1", "1", "2"), credential="oat-token", models=models,
        )

        fetch.assert_awaited_once_with("subscription", "oat-token")
        assert values == {
            "CLAUDE_CODE_OAUTH_TOKEN": "oat-token",
            "CLAUDE_MODEL": "claude-opus-5",
            "CLAUDE_EFFORT": "high",
        }
        assert drops == ["ANTHROPIC_API_KEY"]

    def test_api_key_with_a_model_without_effort(self):
        models = [_model_info("claude-haiku-4-5")]
        values, drops, fetch = _wizard(
            answers=("2", "1"), credential="sk-ant-new", models=models,
            existing={"CLAUDE_EFFORT": "max"},
        )

        fetch.assert_awaited_once_with("api_key", "sk-ant-new")
        assert values == {"ANTHROPIC_API_KEY": "sk-ant-new",
                          "CLAUDE_MODEL": "claude-haiku-4-5"}
        assert drops == ["CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_EFFORT"]

    def test_enter_keeps_the_saved_credential_for_the_fetch(self):
        """A blank answer reuses the saved key, and the list is fetched with it."""
        models = [_model_info("claude-sonnet-5")]
        values, _, fetch = _wizard(
            answers=("2", "1"), credential="", models=models,
            existing={"ANTHROPIC_API_KEY": "sk-ant-saved"},
        )

        fetch.assert_awaited_once_with("api_key", "sk-ant-saved")
        assert values["ANTHROPIC_API_KEY"] == "sk-ant-saved"

    def test_fetch_failure_still_saves_the_credential(self):
        values, drops, _ = _wizard(
            answers=("2",), credential="sk-ant-new",
            fetch_error=httpx.ConnectError("no network"),
        )

        assert values == {"ANTHROPIC_API_KEY": "sk-ant-new"}
        assert drops == ["CLAUDE_CODE_OAUTH_TOKEN"]
