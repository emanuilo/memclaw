"""Backend protocol for the Memclaw agent.

A backend wraps a specific agent SDK (Claude Agent SDK, Cursor SDK, OpenAI
Agents SDK, ...) and exposes a uniform surface to MemclawAgent. The protocol
is intentionally narrow: it covers one-shot text generation (used for memory
consolidation), a full agentic turn with tool access (used for every user
message), per-chat conversation sessions, and model / effort selection.

Conversations live in the backend's own session, keyed by a ``session_key``
MemclawAgent derives from the chat. MemclawAgent persists the session id a
turn reports and hands it back as ``resume_session_id``, so a restart picks
the conversation up where it left off.

Adding a new backend means implementing this protocol and registering the
class in `memclaw.backends.__init__.REGISTRY`. Nothing else in the project
should need SDK-specific imports.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from rich.console import Console

    from ..config import MemclawConfig
    from ..tools import ToolExecutor
    from .claude_models import ModelInfo


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
    # The backend's native session id for this conversation, or None when the
    # backend has no resumable sessions.
    session_id: str | None = None


@dataclass
class ToolStep:
    """One tool call the agent makes during a turn, for progress messages."""

    index: int    # 1-based, per turn
    name: str     # Memclaw tool name, e.g. "memory_search"
    summary: str  # short human-readable description


ProgressCallback = Callable[[ToolStep], Awaitable[None]]


async def report_tool_step(
    on_tool: ProgressCallback | None, index: int, name: str, args: Any,
) -> None:
    """Call *on_tool* for a tool call; a failing callback never fails the turn."""
    if on_tool is None:
        return
    from ..tools import describe_tool_call  # local import — tools imports a lot

    try:
        await on_tool(ToolStep(index, name, describe_tool_call(name, args)))
    except Exception as exc:
        logger.debug("Progress callback failed: {exc}", exc=exc)


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

    # Whether /model and /effort can switch this backend's model at runtime.
    supports_model_selection: ClassVar[bool]

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
    def status_rows(cls, config: "MemclawConfig") -> list[tuple[str, str]]:
        """``(label, value)`` pairs `memclaw status` shows for this backend.

        Typically the model that will run and any settings that shape it.
        Return an empty list when there is nothing worth showing.
        """
        ...

    @classmethod
    def wizard_setup(
        cls,
        console: "Console",
        existing: dict[str, str],
        *,
        memory_dir: Path | str | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        """Interactively collect this backend's env-var values.

        The wizard calls this *only* when this backend has just been
        selected. Implementations are free to print panels, ask
        sub-questions, etc.

        Args:
            console: Rich console for output / prompts.
            existing: env-var values already loaded from ``.env``.
            memory_dir: Active Memclaw memory directory (from ``--memory-dir``),
                or the default when omitted.

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

    # Lifecycle (optional no-ops for backends without extra setup) ---------
    async def on_agent_start(self, tool_executor: "ToolExecutor") -> None:
        """Called once when MemclawAgent starts (after index sync).

        Backends use this for process-scoped resources (e.g. a local MCP server).
        """
        ...

    async def on_agent_shutdown(self) -> None:
        """Called when MemclawAgent shuts down asynchronously."""
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
        context: str,
        user_message: str,
        tool_executor: "ToolExecutor",
        session_key: str,
        resume_session_id: str | None = None,
        image_b64: str | None = None,
        image_media_type: str = "image/jpeg",
        max_turns: int = 10,
        on_tool: ProgressCallback | None = None,
    ) -> TurnResult:
        """Run one full agentic turn with tool access, in the conversation
        identified by *session_key*.

        ``system_prompt`` is static per session: it is only used when the
        backend starts a new session. ``context`` is the per-turn context
        block (time, relevant memories, ...) that must reach the model with
        this message, ahead of ``user_message``. ``resume_session_id`` is
        the session to resume if the backend has to (re)connect; if it can't
        be resumed the backend starts a fresh one.

        ``on_tool`` is called (via `report_tool_step`) for every tool call
        the agent makes, for progress messages.

        The backend is responsible for translating ``TOOL_DEFINITIONS``
        into its SDK's tool format and routing tool calls back through
        ``tool_executor.execute(name, args)``.

        ``max_turns`` is passed to SDKs that support it (e.g. Claude).
        Backends without a native cap should log when reported turns exceed
        this limit after the run completes.
        """
        ...

    async def reset_session(self, session_key: str) -> None:
        """Forget the conversation for *session_key* (the `/new` command)."""
        ...

    # Model selection (only used when supports_model_selection) ------------
    @property
    def model(self) -> str:
        """The model the next turn will run on."""
        ...

    @property
    def effort(self) -> str | None:
        """The effort level the next turn will run with (None: model default)."""
        ...

    async def list_models(self) -> list["ModelInfo"]:
        """Models the configured credential can use. Raises on failure."""
        ...

    async def effort_levels(self) -> list[str] | None:
        """Effort levels the current model accepts ([]: none, None: unknown)."""
        ...

    async def set_model(self, model_id: str) -> str:
        """Switch model, keeping conversations. Returns a note for the user
        (e.g. when the effort level had to change). Raises ValueError for
        an unknown model."""
        ...

    async def set_effort(self, level: str) -> None:
        """Set the effort level. Raises ValueError if the model rejects it."""
        ...


# Subclasses register themselves through this attribute name; see
# memclaw/backends/__init__.py.
__all__ = ["AgentBackend", "TurnResult"]
