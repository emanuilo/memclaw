"""Agent-backend registry.

Each backend module (e.g. `claude.py`) exposes a class that satisfies the
`AgentBackend` protocol. To register one, add it to `REGISTRY` below.

Selection at runtime is driven by the `AGENT_BACKEND` env var (mirrored on
`MemclawConfig.agent_backend`). The default is `claude`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentBackend, TurnRequest, TurnResult
from .claude import ClaudeAgentBackend
from .cursor import CursorAgentBackend

if TYPE_CHECKING:
    from ..config import MemclawConfig


DEFAULT_BACKEND = "claude"

REGISTRY: dict[str, type[AgentBackend]] = {
    ClaudeAgentBackend.name: ClaudeAgentBackend,
    CursorAgentBackend.name: CursorAgentBackend,
}


def list_backends() -> list[type[AgentBackend]]:
    return list(REGISTRY.values())


def get_backend_class(name: str) -> type[AgentBackend]:
    """Look up a backend class by short name. Raises ValueError if unknown."""
    cls = REGISTRY.get(name)
    if cls is None:
        known = ", ".join(REGISTRY) or "(none)"
        raise ValueError(f"Unknown agent backend {name!r}. Available: {known}")
    return cls


def resolve_backend_name(config: "MemclawConfig") -> str:
    """Return the backend name the config selects, falling back to default."""
    return config.agent_backend or DEFAULT_BACKEND


def build_backend(config: "MemclawConfig") -> AgentBackend:
    """Instantiate the backend chosen by *config*."""
    return get_backend_class(resolve_backend_name(config))(config)


def configuration_problem(config: "MemclawConfig") -> str | None:
    """Why the backend *config* chooses can't run (an unknown name, missing
    credentials), as a message for the user with Rich markup; None if it can."""
    name = resolve_backend_name(config)
    backend_cls = REGISTRY.get(name)
    if backend_cls is None:
        known = ", ".join(REGISTRY) or "(none)"
        return (
            f"unknown agent backend [bold]{name}[/bold] "
            f"(set via AGENT_BACKEND). Available: {known}.\n"
            "Run [bold]memclaw configure[/bold] to pick a valid backend."
        )
    if backend_cls.is_configured(config):
        return None
    return f"{backend_cls.configuration_help()}\nRun [bold]memclaw configure[/bold] to set it."


def status_rows(config: "MemclawConfig") -> list[tuple[str, str]]:
    """``(label, value)`` pairs `memclaw status` shows for the chosen backend."""
    return get_backend_class(resolve_backend_name(config)).status_rows(config)


__all__ = [
    "AgentBackend",
    "ClaudeAgentBackend",
    "CursorAgentBackend",
    "DEFAULT_BACKEND",
    "REGISTRY",
    "TurnRequest",
    "TurnResult",
    "build_backend",
    "configuration_problem",
    "get_backend_class",
    "list_backends",
    "resolve_backend_name",
    "status_rows",
]
