"""Agent-backend registry.

Each backend module (e.g. `claude.py`) exposes a class that satisfies the
`AgentBackend` protocol. To register one, add it to `REGISTRY` below.

Selection at runtime is driven by the `AGENT_BACKEND` env var (mirrored on
`MemclawConfig.agent_backend`). The default is `claude`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentBackend, TurnResult
from .claude import ClaudeAgentBackend

if TYPE_CHECKING:
    from ..config import MemclawConfig


DEFAULT_BACKEND = "claude"

REGISTRY: dict[str, type] = {
    ClaudeAgentBackend.name: ClaudeAgentBackend,
}


def list_backends() -> list[type]:
    return list(REGISTRY.values())


def get_backend_class(name: str) -> type:
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


__all__ = [
    "AgentBackend",
    "ClaudeAgentBackend",
    "DEFAULT_BACKEND",
    "REGISTRY",
    "TurnResult",
    "build_backend",
    "get_backend_class",
    "list_backends",
    "resolve_backend_name",
]
