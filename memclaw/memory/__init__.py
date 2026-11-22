"""Memclaw's memory subsystem: the notes, how they are searched, and
MEMORY.md.

Public:

- `MemoryStore` (notes.py): writes and lists the Markdown notes, the daily
  files under memory/ that are the complete record.
- `HybridSearch`, `SearchResult` (search.py): ranks chunks of the notes for
  a query (vector + keyword, recency decay, near-duplicates dropped).
- `Consolidator` (consolidation.py): distills the daily notes into
  MEMORY.md, and is the only code that writes it. refs.py checks the
  references it writes.

Other code imports from `memclaw.memory`, not from the modules inside.
"""

from .consolidation import Consolidator
from .notes import MemoryStore
from .search import HybridSearch, SearchResult

__all__ = ["Consolidator", "HybridSearch", "MemoryStore", "SearchResult"]
