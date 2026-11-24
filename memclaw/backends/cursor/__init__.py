"""The Cursor Python SDK backend.

`backend` is the `AgentBackend` itself. Each chat's agent (`agents`) runs
on one shared bridge process (`bridge`); `sdk_adapter` reads a run's event
stream; `hooks` restricts the agent to Memclaw's tools; `configure` asks
the setup questions and installs the hooks.
"""

from .backend import CursorAgentBackend

__all__ = ["CursorAgentBackend"]
