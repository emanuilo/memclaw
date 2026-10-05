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


class TestRunTurn:
    @pytest.mark.asyncio
    async def test_returns_turn_result(self, claude_config):
        from memclaw.tools import ToolExecutor

        backend = ClaudeAgentBackend(claude_config(api_key="x"))
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

    @pytest.mark.asyncio
    async def test_fallback_cost_uses_the_configured_models_price(self, claude_config):
        from memclaw.tools import ToolExecutor

        cfg = claude_config(api_key="x")
        cfg.claude_model = "claude-opus-5"
        backend = ClaudeAgentBackend(cfg)
        executor = ToolExecutor(
            config=cfg, store=MagicMock(), index=MagicMock(),
            search=MagicMock(), found_images=[], platform="test",
        )

        ctx_factory, _client = _mock_sdk_client("done")
        with patch("memclaw.backends.claude.ClaudeSDKClient", side_effect=ctx_factory):
            result = await backend.run_turn(
                system_prompt="sys", user_message="hello", tool_executor=executor,
            )

        # 100 input tokens at $5/M + 50 output tokens at $25/M (Opus 5).
        assert result.cost_usd == pytest.approx((100 * 5 + 50 * 25) / 1_000_000)


class TestPrices:
    @pytest.mark.parametrize("model, prices", [
        ("claude-opus-5", (5.0, 25.0)),
        ("claude-opus-5-5", (4.0, 20.0)),         # not mistaken for claude-opus-5
        ("claude-sonnet-4-6", (3.0, 15.0)),
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
    def test_unset_gives_none(self, claude_config):
        cfg = claude_config(oauth="x")
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
        assert rows == [("Model", claude_backend._DEFAULT_MODEL), ("Effort", "default")]


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

        assert seen["options"].model == claude_backend._DEFAULT_MODEL
        assert seen["options"].effort is None


# ────────────────────────────────────────────────────────────────────
# Wizard: model + effort questions
# ────────────────────────────────────────────────────────────────────

def _model_info(model_id: str, *, effort_levels: list[str] | None = None):
    return ModelInfo(
        id=model_id,
        display_name=model_id,
        created_at="2026-07-24T00:00:00Z",
        effort_levels=effort_levels or [],
    )


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

    def test_effort_defaults_to_high(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "medium", "high"])]
        _, _, prompts, _ = _ask(models, answers=("1", "3"))
        assert prompts[1]["default"] == "3"

    def test_effort_default_falls_back_when_high_is_unsupported(self):
        models = [_model_info("claude-opus-5", effort_levels=["low", "medium"])]
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
