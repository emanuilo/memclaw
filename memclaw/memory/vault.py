"""The notes and their index together, for the one-off `memclaw` commands.

`memclaw save`, `search`, `index`, `consolidate` and `status` each open the
notes and the index, do one thing and close them again: one `MemoryVault`
method per command. (The agent keeps its own store and index open for as
long as it runs.)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from loguru import logger

from ..config import MemclawConfig
from .consolidation import Consolidator, OneShotBackend
from .index import MemoryIndex
from .notes import MemoryStore
from .search import HybridSearch, SearchResult


class MemoryVault:
    """The notes (`store`) and their search index (`index`); use it as a
    context manager, which closes the index."""

    def __init__(self, config: MemclawConfig):
        self.config = config
        self.store = MemoryStore(config)
        self.index = MemoryIndex(config)

    def __enter__(self) -> MemoryVault:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.index.close()

    async def save(self, content: str) -> Path:
        """Add *content* to today's daily note and index it. Returns the
        daily file."""
        file_path = self.store.save(content)
        await self.index.index_file(file_path)
        return file_path

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """The notes' chunks that best match *query* (see HybridSearch)."""
        return await HybridSearch(self.config, self.index).search(query, limit=limit)

    async def reindex(self) -> tuple[bool, dict[str, int]]:
        """Re-index the notes that changed. Returns whether anything did, and
        the index's counts (chunks, files, images)."""
        changed = await self.index.sync()
        return changed, self.index.get_stats()

    async def sync_or_warn(self) -> None:
        """Re-index the notes that changed; a failure (e.g. OpenAI
        unreachable) is logged, not raised."""
        try:
            await self.index.sync()
        except Exception as exc:
            logger.warning("Index sync at startup failed: {exc}", exc=exc)

    async def consolidate(self, backend: OneShotBackend, *, since: date | None = None) -> bool:
        """Distill the daily notes into MEMORY.md now, with *backend*'s LLM.
        Returns False if there was nothing to consolidate."""
        consolidator = Consolidator(self.config, self.store, self.index, backend)
        try:
            return await consolidator.consolidate(force=True, since=since)
        finally:
            await consolidator.aclose()
