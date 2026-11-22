"""Memclaw's memory subsystem: the notes, the index and search over them,
and MEMORY.md.

Public:

- `MemoryStore` (notes.py): writes and lists the Markdown notes, the daily
  files under memory/ that are the complete record.
- `MemoryIndex` (index.py): the SQLite index of the notes (chunks,
  embeddings, FTS5) and of saved images. The only SQLite user here;
  chunking.py splits the notes for it.
- `HybridSearch`, `SearchResult` (search.py): ranks chunks of the notes for
  a query (vector + keyword, recency decay, near-duplicates dropped),
  reading the index through MemoryIndex's methods.
- `Consolidator` (consolidation.py): distills the daily notes into
  MEMORY.md, and is the only code that writes it. refs.py checks the
  references it writes.
- `MemoryVault` (vault.py): the notes and the index opened together, one
  method per one-off `memclaw` command (save, search, index, consolidate).

Other code imports from `memclaw.memory`, not from the modules inside.
"""

from .consolidation import Consolidator
from .index import MemoryIndex
from .notes import MemoryStore
from .search import HybridSearch, SearchResult
from .vault import MemoryVault

__all__ = [
    "Consolidator",
    "HybridSearch",
    "MemoryIndex",
    "MemoryStore",
    "MemoryVault",
    "SearchResult",
]
