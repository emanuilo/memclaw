"""Memclaw's memory subsystem.

Public: `Consolidator`, which distills the daily notes into MEMORY.md (and
is the only code that writes it). Other code imports from `memclaw.memory`,
not from the modules inside.
"""

from .consolidation import Consolidator

__all__ = ["Consolidator"]
