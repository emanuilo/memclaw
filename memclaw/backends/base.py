"""Backend protocol for the Memclaw agent.

A backend wraps a specific agent SDK (Claude Agent SDK, Cursor SDK, OpenAI
Agents SDK, ...) and exposes a uniform surface to MemclawAgent. The protocol
is intentionally narrow: it covers exactly the two interactions MemclawAgent
needs — one-shot text generation (used for memory consolidation) and a full
agentic turn with tool access (used for every user message).

Adding a new backend means implementing this protocol and registering the
class in `memclaw.backends.__init__.REGISTRY`. Nothing else in the project
should need SDK-specific imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor


@dataclass
class TurnResult:
    """Normalized result of one user-facing agent turn.

    Token fields are reported when the SDK surfaces them; backends that
    don't expose usage data should leave them at 0. `cost_usd` is None
    when the backend doesn't compute per-call cost (e.g. subscription
    billing where requests are paid against a plan, not per token).
    """

    text: str
    num_turns: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None


@runtime_checkable
class AgentBackend(Protocol):
    """The contract every agent backend must satisfy."""

    # Identity ---------------------------------------------------------
    name: ClassVar[str]          # short identifier, e.g. "claude"
    display_name: ClassVar[str]  # human label, e.g. "Claude Agent SDK"

    # Billing semantics drive whether the per-turn cost line is shown.
    # True for pay-per-token (API key); False when usage is bundled into
    # a subscription/plan.
    bills_per_token: bool

    def __init__(self, config: "MemclawConfig") -> None: ...

    # Configuration --------------------------------------------------------
    @classmethod
    def is_configured(cls, config: "MemclawConfig") -> bool:
        """Return True if *config* carries enough credentials to run."""
        ...

    @classmethod
    def configuration_help(cls) -> str:
        """Multi-line text shown when `is_configured` returns False."""
        ...

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        """Interactively collect this backend's env-var values.

        The wizard calls this *only* when this backend has just been
        selected. Implementations are free to print panels, ask
        sub-questions, etc.

        Args:
            console: Rich console for output / prompts.
            existing: env-var values already loaded from ``.env``.

        Returns:
            A `(values, drop_keys)` pair.
            - ``values`` maps env-var name → user-provided value for keys
              this backend wants saved.
            - ``drop_keys`` lists env-var names this backend wants removed
              from both saved config and the live process environment
              (used to scrub credentials from a previously-selected backend
              so they can't shadow the new choice).
        """
        ...

    # Runtime --------------------------------------------------------------
    async def run_one_shot(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Single-turn, tool-free LLM call. Returns the response text."""
        ...

    async def run_turn(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tool_executor: "ToolExecutor",
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
    ) -> TurnResult:
        """Run one full agentic turn with tool access.

        The backend is responsible for translating ``TOOL_DEFINITIONS``
        into its SDK's tool format, routing tool calls back through
        ``tool_executor.execute(name, args)``, and capping iterations
        at ``max_turns``.
        """
        ...


# Subclasses register themselves through this attribute name; see
# memclaw/backends/__init__.py.
__all__ = ["AgentBackend", "TurnResult"]
