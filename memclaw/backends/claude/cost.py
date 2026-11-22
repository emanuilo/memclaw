"""What a Claude turn costs: the CLI's running session total turned into a
per-turn figure, or an estimate from the token counts."""

from __future__ import annotations

from ..base import TurnResult
from .settings import DEFAULT_MODEL

# Anthropic list prices, (input, output) USD per 1M tokens. Only used as a
# fallback when the SDK doesn't return total_cost_usd for an API-key turn.
# Ids are matched by prefix, so dated snapshots (claude-haiku-4-5-20251001)
# resolve too. A model missing here (an alias, or one released later) is
# estimated at the default model's price, so the figure is approximate.
_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Cache reads are billed at a tenth of the input price, cache writes at
# 1.25 times it.
_CACHE_READ_MULTIPLIER = 0.1
_CACHE_WRITE_MULTIPLIER = 1.25


def _prices_per_m(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for *model*; see _PRICES_PER_M."""
    # Longest prefix first, so claude-opus-5-5 doesn't match claude-opus-5.
    for prefix in sorted(_PRICES_PER_M, key=len, reverse=True):
        if model == prefix or model.startswith(f"{prefix}-"):
            return _PRICES_PER_M[prefix]
    return _PRICES_PER_M[DEFAULT_MODEL]


class SessionCosts:
    """Turns the running cost total the CLI reports per session into each
    turn's cost."""

    def __init__(self, *, bills_per_token: bool) -> None:
        self._bills_per_token = bills_per_token
        # Last total_cost_usd seen per session id. The CLI reports a running
        # total for the session (carried over on resume), not a per-turn cost.
        self._totals: dict[str, float] = {}

    def turn_cost(
        self, result: TurnResult, resume_session_id: str | None, *, model: str,
    ) -> float | None:
        """This turn's cost from the session's running total, which is what
        the CLI reports. Without a baseline (the first turn after a restart
        resumed the session) it is estimated from the token counts instead."""
        total = result.cost_usd
        if total is not None and result.session_id:
            previous = self._totals.get(result.session_id)
            self._totals[result.session_id] = total
            if previous is not None or resume_session_id != result.session_id:
                return max(total - (previous or 0.0), 0.0)
        if not self._bills_per_token:
            return None
        input_per_m, output_per_m = _prices_per_m(model)
        return (
            result.input_tokens * input_per_m
            + result.output_tokens * output_per_m
            + result.cache_read_tokens * input_per_m * _CACHE_READ_MULTIPLIER
            + result.cache_creation_tokens * input_per_m * _CACHE_WRITE_MULTIPLIER
        ) / 1_000_000
