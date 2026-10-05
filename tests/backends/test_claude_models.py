"""Tests for the Anthropic /v1/models discovery module.

Every HTTP call is mocked — the suite runs with no API keys and no network.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memclaw.backends import claude_models
from memclaw.backends.claude_models import fetch_models


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _effort(*levels: str, supported: bool = True) -> dict:
    """Build an `effort` capability node advertising *levels*."""
    node: dict = {"supported": supported}
    for level in ("low", "medium", "high", "xhigh", "max"):
        node[level] = {"supported": level in levels}
    return node


def _entry(model_id: str, created_at: str, *, effort: dict | None = None,
           display_name: str | None = None) -> dict:
    return {
        "id": model_id,
        "display_name": display_name or model_id,
        "created_at": created_at,
        "capabilities": {"effort": effort} if effort is not None else {},
    }


def _mock_httpx(payload: dict | None = None, *, calls: list | None = None,
                get_error: Exception | None = None,
                status_error: Exception | None = None):
    """Patch httpx.AsyncClient with a fake that records the request it got.

    Returns the patcher; use it as a context manager.
    """
    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=status_error)
    response.json = MagicMock(return_value=payload or {"data": []})

    async def _get(url, headers=None, params=None):
        if calls is not None:
            calls.append({"url": url, "headers": headers, "params": params})
        if get_error is not None:
            raise get_error
        return response

    client = MagicMock()
    client.get = _get

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)

    return patch("memclaw.backends.claude_models.httpx.AsyncClient",
                 MagicMock(return_value=ctx))


# ────────────────────────────────────────────────────────────────────
# Authentication headers
# ────────────────────────────────────────────────────────────────────

class TestAuthHeaders:
    @pytest.mark.asyncio
    async def test_api_key_sends_x_api_key(self):
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models("api_key", "sk-ant-test")

        headers = calls[0]["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_oauth_sends_bearer_not_x_api_key(self):
        """An OAuth token sent as x-api-key gets a 401 — it must never happen."""
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models("subscription", "oat-token")

        headers = calls[0]["headers"]
        assert headers["Authorization"] == "Bearer oat-token"
        assert "x-api-key" not in headers
        assert "anthropic-beta" not in headers   # /v1/models doesn't need one

    @pytest.mark.asyncio
    async def test_api_version_always_sent(self):
        calls: list = []
        with _mock_httpx(calls=calls):
            await fetch_models("api_key", "sk-ant-test")

        assert calls[0]["headers"]["anthropic-version"] == claude_models.API_VERSION

    @pytest.mark.asyncio
    async def test_no_credential_raises(self):
        with _mock_httpx(), pytest.raises(RuntimeError, match="No Claude credential"):
            await fetch_models("api_key", "")

    @pytest.mark.asyncio
    async def test_unknown_auth_mode_raises(self):
        with _mock_httpx(), pytest.raises(RuntimeError, match="No Claude credential"):
            await fetch_models("", "sk-ant-test")


# ────────────────────────────────────────────────────────────────────
# Parsing the capability tree
# ────────────────────────────────────────────────────────────────────

class TestEffortLevels:
    @pytest.mark.asyncio
    async def test_supported_levels_come_back_in_sdk_order(self):
        payload = {"data": [
            _entry("claude-opus-5", "2026-07-24T00:00:00Z",
                   effort=_effort("low", "medium", "high", "xhigh", "max")),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].effort_levels == ["low", "medium", "high", "xhigh", "max"]

    @pytest.mark.asyncio
    async def test_partial_support_keeps_only_supported_levels(self):
        payload = {"data": [
            _entry("claude-sonnet-5", "2026-05-01T00:00:00Z",
                   effort=_effort("low", "high")),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].effort_levels == ["low", "high"]

    @pytest.mark.asyncio
    async def test_model_without_effort_support_has_no_levels(self):
        """Haiku 4.5 reports effort.supported=false — never offer it a level."""
        payload = {"data": [
            _entry("claude-haiku-4-5", "2025-10-01T00:00:00Z",
                   effort=_effort(supported=False)),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].effort_levels == []

    @pytest.mark.asyncio
    async def test_missing_effort_node_has_no_levels(self):
        payload = {"data": [_entry("claude-old", "2024-01-01T00:00:00Z")]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].effort_levels == []

    @pytest.mark.asyncio
    async def test_level_the_sdk_does_not_know_is_dropped(self):
        """A level Anthropic ships before our SDK understands it must not leak."""
        effort = _effort("high")
        effort["ultra"] = {"supported": True}  # hypothetical future level
        payload = {"data": [_entry("claude-future", "2027-01-01T00:00:00Z", effort=effort)]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].effort_levels == ["high"]


# ────────────────────────────────────────────────────────────────────
# Ordering and shape
# ────────────────────────────────────────────────────────────────────

class TestParsing:
    @pytest.mark.asyncio
    async def test_sorted_newest_first(self):
        payload = {"data": [
            _entry("old", "2024-01-01T00:00:00Z"),
            _entry("newest", "2026-07-24T00:00:00Z"),
            _entry("middle", "2025-06-01T00:00:00Z"),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert [m.id for m in models] == ["newest", "middle", "old"]

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_sorts_last(self):
        payload = {"data": [
            _entry("broken", "not-a-date"),
            _entry("fine", "2025-01-01T00:00:00Z"),
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert [m.id for m in models] == ["fine", "broken"]

    @pytest.mark.asyncio
    async def test_display_name_falls_back_to_id(self):
        payload = {"data": [
            {"id": "claude-bare", "created_at": "2026-01-01T00:00:00Z"},
        ]}
        with _mock_httpx(payload):
            models = await fetch_models("api_key", "k")

        assert models[0].display_name == "claude-bare"

    @pytest.mark.asyncio
    async def test_empty_data_gives_empty_list(self):
        with _mock_httpx({"data": []}):
            models = await fetch_models("api_key", "k")

        assert models == []


# ────────────────────────────────────────────────────────────────────
# Failures propagate — the caller decides what to do
# ────────────────────────────────────────────────────────────────────

class TestFailures:
    @pytest.mark.asyncio
    async def test_network_error_propagates(self):
        with _mock_httpx(get_error=httpx.ConnectError("no network")):
            with pytest.raises(httpx.ConnectError):
                await fetch_models("api_key", "k")

    @pytest.mark.asyncio
    async def test_http_status_error_propagates(self):
        error = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(),
        )
        with _mock_httpx(status_error=error):
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_models("api_key", "bad-key")


class TestNoCache:
    @pytest.mark.asyncio
    async def test_every_call_fetches_fresh(self):
        """No cache: a new credential must never see the previous one's list."""
        calls: list = []
        with _mock_httpx({"data": [_entry("m", "2026-01-01T00:00:00Z")]}, calls=calls):
            await fetch_models("api_key", "first-key")
            await fetch_models("subscription", "second-token")

        assert len(calls) == 2
        assert calls[1]["headers"]["Authorization"] == "Bearer second-token"
