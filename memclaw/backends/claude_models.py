"""Discover which Claude models the user's credential can actually reach.

Anthropic's /v1/models endpoint returns every model available to the
configured credential, each with a capability tree describing which effort
levels it accepts. The setup wizard builds its model picker from this, so a
model Anthropic releases shows up with no code change here.

The list is fetched fresh on every call. It is only needed by
`memclaw configure`, and a cached copy would go stale the moment the user
switches to a credential that can reach a different set of models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

MODELS_URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"

REQUEST_TIMEOUT = 10.0

# The endpoint pages at 20 by default; ask for more so a long model list
# doesn't get silently truncated.
PAGE_LIMIT = 100

# Effort levels claude-agent-sdk accepts — ClaudeAgentOptions.effort is
# Optional[Literal["low", "medium", "high", "xhigh", "max"]]. The API may
# report a level a newer SDK understands and ours does not, so anything
# outside this set is dropped. Widen the tuple when the SDK gains a level.
SDK_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class ModelInfo:
    id: str
    display_name: str
    created_at: str
    effort_levels: list[str]  # empty when the model has no effort support


# ── Request building ────────────────────────────────────────────────

def _build_headers(auth_mode: str, credential: str) -> dict[str, str]:
    """Pick the auth header matching the credential type.

    *auth_mode* is "subscription" or "api_key", as returned by
    `_claude_auth_mode`. The two credential types are not interchangeable:
    an OAuth token sent as `x-api-key` comes back 401, so these cases must
    stay separate.
    """
    if not credential:
        raise RuntimeError("No Claude credential is configured.")
    headers = {"anthropic-version": API_VERSION}
    if auth_mode == "subscription":
        headers["Authorization"] = f"Bearer {credential}"
    elif auth_mode == "api_key":
        headers["x-api-key"] = credential
    else:
        raise RuntimeError("No Claude credential is configured.")
    return headers


# ── Response parsing ────────────────────────────────────────────────

def _effort_levels(capabilities: dict[str, Any]) -> list[str]:
    """Read the supported effort levels off one model's capability tree.

    Iterating SDK_EFFORT_LEVELS rather than the API's own keys is what makes
    an unknown future level drop out on its own.
    """
    effort = capabilities.get("effort")
    if not isinstance(effort, dict) or not effort.get("supported"):
        return []
    return [
        level
        for level in SDK_EFFORT_LEVELS
        if isinstance(effort.get(level), dict) and effort[level].get("supported")
    ]


def _created_key(created_at: str) -> datetime:
    """Parse an ISO-8601 timestamp into something sortable.

    Python 3.10's fromisoformat rejects a trailing "Z", and aware and naive
    datetimes can't be compared to each other, so both are normalised here.
    A value we can't parse sorts last instead of breaking the whole list.
    """
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_models(payload: dict[str, Any]) -> list[ModelInfo]:
    models = [
        ModelInfo(
            id=entry["id"],
            display_name=entry.get("display_name") or entry["id"],
            created_at=entry.get("created_at") or "",
            effort_levels=_effort_levels(entry.get("capabilities") or {}),
        )
        for entry in payload.get("data") or []
        if entry.get("id")
    ]
    models.sort(key=lambda m: _created_key(m.created_at), reverse=True)
    return models


# ── Entry point ─────────────────────────────────────────────────────

async def fetch_models(auth_mode: str, credential: str) -> list[ModelInfo]:
    """Fetch models from /v1/models, newest first.

    Raises on network or auth failure — the caller decides what to do.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            MODELS_URL,
            headers=_build_headers(auth_mode, credential),
            params={"limit": PAGE_LIMIT},
        )
        response.raise_for_status()
        return _parse_models(response.json())
