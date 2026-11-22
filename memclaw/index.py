from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from loguru import logger

from .chunking import Chunk, chunk_text
from .config import MemclawConfig
from .openai_client import OpenAIClient


def _create_schema(db: sqlite3.Connection) -> None:
    """Create the tables, the FTS5 index and its triggers if missing.

    SQLite stores each statement's text, indentation included, in
    sqlite_master: keep the text unchanged so every database has the same
    schema.
    """
    db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                file_mtime REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

    db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

    # Triggers keep FTS5 in sync with the chunks table.
    db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
    db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END
        """)
    db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

    db.execute("""
            CREATE TABLE IF NOT EXISTS file_meta (
                file_path TEXT PRIMARY KEY,
                mtime REAL NOT NULL
            )
        """)

    # Embedding cache — avoids redundant OpenAI API calls for unchanged content.
    db.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding    BLOB NOT NULL,
                model        TEXT NOT NULL,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

    # Telegram image registry — maps file_ids to descriptions for retrieval.
    db.execute("""
            CREATE TABLE IF NOT EXISTS telegram_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                description TEXT NOT NULL,
                caption TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

    # Platform-agnostic image registry (Telegram, WhatsApp, Slack, etc.)
    db.execute("""
            CREATE TABLE IF NOT EXISTS platform_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                media_ref TEXT NOT NULL,
                description TEXT NOT NULL,
                caption TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

    _migrate_telegram_images(db)


def _migrate_telegram_images(db: sqlite3.Connection) -> None:
    """One-time migration: copy telegram_images rows into platform_images."""
    existing = db.execute(
        "SELECT COUNT(*) FROM platform_images WHERE platform = 'telegram'"
    ).fetchone()[0]
    if existing > 0:
        return
    telegram_count = db.execute("SELECT COUNT(*) FROM telegram_images").fetchone()[0]
    if telegram_count == 0:
        return
    db.execute("""
            INSERT INTO platform_images (platform, media_ref, description, caption, embedding, created_at)
            SELECT 'telegram', file_id, description, caption, embedding, created_at
            FROM telegram_images
        """)


def _image_dict(row: tuple, score: float | None = None) -> dict:
    """(id, platform, media_ref, description, caption, created_at, ...)
    as the dict the bots send images from."""
    return {
        "id": row[0],
        "platform": row[1],
        "media_ref": row[2],
        "file_id": row[2],  # backward compat alias
        "description": row[3],
        "caption": row[4],
        "created_at": row[5],
        "score": score,
    }


class MemoryIndex:
    """SQLite-based index with FTS5 keyword search and numpy vector search.

    Embeddings are stored as BLOBs in SQLite and cosine similarity is computed
    in-memory with numpy — lightweight and dependency-free beyond numpy.
    """

    def __init__(self, config: MemclawConfig):
        self.config = config
        self._openai: OpenAIClient | None = None
        self._init_db()

    def _init_db(self):
        self.db = sqlite3.connect(str(self.config.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        _create_schema(self.db)
        self.db.commit()

    @property
    def openai(self) -> OpenAIClient:
        if self._openai is None:
            self._openai = OpenAIClient(self.config.openai.api_key)
        return self._openai

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        model = self.config.memory.embedding_model

        # Compute content hashes and check cache
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, h in enumerate(hashes):
            row = self.db.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = ? AND model = ?",
                (h, model),
            ).fetchone()
            if row is not None:
                results[i] = self.deserialize_embedding(row[0])
            else:
                uncached_indices.append(i)

        # Fetch embeddings for uncached texts only
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            vectors = await self.openai.embed(uncached_texts, model=model)
            new_embeddings = [np.array(v, dtype=np.float32) for v in vectors]

            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                self.db.execute(
                    "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, model) VALUES (?, ?, ?)",
                    (hashes[idx], self.serialize_embedding(emb), model),
                )
            self.db.commit()

        return results  # type: ignore[return-value]

    async def get_embedding(self, text: str) -> np.ndarray:
        results = await self.get_embeddings([text])
        return results[0]

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def is_indexable(self, path: Path) -> bool:
        """Whether *path* holds memories: any ``.md`` under the memory dir
        except AGENTS.md (instructions, already in the system prompt) and
        files in hidden directories."""
        try:
            rel = Path(path).resolve().relative_to(self.config.memory_dir.resolve())
        except ValueError:
            return False
        if rel.suffix != ".md" or rel == Path(self.config.agent_file.name):
            return False
        return not any(part.startswith(".") for part in rel.parts)

    def indexable_files(self) -> list[Path]:
        return sorted(
            p for p in self.config.memory_dir.glob("**/*.md")
            if p.is_file() and self.is_indexable(p)
        )

    async def index_file(self, file_path: Path):
        """Re-index a single file: chunk, embed, store.

        Embeddings are fetched before anything is replaced, so a failed
        embedding call leaves the file's previous chunks in place.
        """
        content = file_path.read_text()
        mtime = file_path.stat().st_mtime
        path_str = str(file_path)

        memory = self.config.memory
        chunks = chunk_text(
            content, path_str,
            target_words=memory.chunk_target_words, overlap_words=memory.chunk_overlap_words,
        ) if content.strip() else []
        embeddings = await self.get_embeddings([c.content for c in chunks])

        with self.db:
            self.db.execute("DELETE FROM chunks WHERE file_path = ?", (path_str,))
            for chunk, emb in zip(chunks, embeddings):
                self.db.execute(
                    """INSERT INTO chunks
                       (file_path, line_start, line_end, content, embedding, file_mtime)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.file_path,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.content,
                        self.serialize_embedding(emb),
                        mtime,
                    ),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO file_meta (file_path, mtime) VALUES (?, ?)",
                (path_str, mtime),
            )

    async def sync(self) -> bool:
        """Re-index any memory files that have changed since last indexed.

        Returns True if anything was updated.
        """
        changed = False
        files = self.indexable_files()
        for md_file in files:
            mtime = md_file.stat().st_mtime
            row = self.db.execute(
                "SELECT mtime FROM file_meta WHERE file_path = ?",
                (str(md_file),),
            ).fetchone()
            if row is None or mtime != row[0]:
                await self.index_file(md_file)
                changed = True

        # Remove entries for deleted (or no longer indexable) files
        stored = {r[0] for r in self.db.execute("SELECT file_path FROM file_meta").fetchall()}
        stored |= {r[0] for r in self.db.execute("SELECT DISTINCT file_path FROM chunks").fetchall()}
        existing = {str(f) for f in files}
        for gone in stored - existing:
            self.db.execute("DELETE FROM chunks WHERE file_path = ?", (gone,))
            self.db.execute("DELETE FROM file_meta WHERE file_path = ?", (gone,))
            changed = True

        if changed:
            self.db.commit()
        return changed

    # ------------------------------------------------------------------
    # Queries for HybridSearch
    # ------------------------------------------------------------------

    def embeddings_version(self) -> tuple[int, int]:
        """(count, highest id) of the chunks that have an embedding. It
        changes whenever chunks change (re-indexing a file gives its rows
        new, higher ids), so a search can cache the embeddings until then."""
        count, max_id = self.db.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        return count, max_id

    def chunk_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """(chunk id, embedding) of every chunk that has one."""
        rows = self.db.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(chunk_id, self.deserialize_embedding(blob)) for chunk_id, blob in rows]

    def keyword_matches(self, fts_query: str, limit: int | None) -> list[tuple[int, float]]:
        """(chunk id, FTS5 rank) of the best *limit* matches (all with None)
        for an FTS5 query, best first. FTS5's rank is BM25 negated: more
        negative is a better match. Empty if the query fails."""
        try:
            return self.db.execute(
                """SELECT rowid, rank FROM chunks_fts
                   WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (fts_query, -1 if limit is None else limit),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("Keyword search failed for {q!r}: {exc}", q=fts_query, exc=exc)
            return []

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        """The stored chunk with this id (without its embedding)."""
        row = self.db.execute(
            "SELECT file_path, line_start, line_end, content FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        return Chunk(*row) if row else None

    # ------------------------------------------------------------------
    # Platform-agnostic image registry
    # ------------------------------------------------------------------

    async def store_platform_image(
        self,
        platform: str,
        media_ref: str,
        description: str,
        caption: str | None = None,
    ):
        """Store an image reference for any platform (telegram, whatsapp, slack, etc.)."""
        embedding = await self.get_embedding(description)
        self.db.execute(
            "INSERT INTO platform_images (platform, media_ref, description, caption, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (platform, media_ref, description, caption, self.serialize_embedding(embedding)),
        )
        self.db.commit()

    def search_platform_images(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        platform: str | None = None,
    ) -> list[dict]:
        """Vector search over stored images. Optionally filter by platform."""
        if platform:
            rows = self.db.execute(
                "SELECT id, platform, media_ref, description, caption, created_at, embedding "
                "FROM platform_images WHERE embedding IS NOT NULL AND platform = ?",
                (platform,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, platform, media_ref, description, caption, created_at, embedding "
                "FROM platform_images WHERE embedding IS NOT NULL"
            ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            stored_emb = self.deserialize_embedding(row[6])
            similarity = float(
                np.dot(query_embedding, stored_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb) + 1e-8)
            )
            results.append(_image_dict(row, score=similarity))

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_platform_images(self, ids: list[int], platform: str | None = None) -> list[dict]:
        """Stored images by id, in the order given; ids that don't exist
        (or belong to another platform) are left out."""
        found = {}
        for image_id in ids:
            row = self.db.execute(
                "SELECT id, platform, media_ref, description, caption, created_at "
                "FROM platform_images WHERE id = ? AND (? IS NULL OR platform = ?)",
                (image_id, platform, platform),
            ).fetchone()
            if row is not None:
                found[row[0]] = _image_dict(row)
        return list(found.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = self.db.execute("SELECT COUNT(*) FROM file_meta").fetchone()[0]
        images = self.db.execute("SELECT COUNT(*) FROM platform_images").fetchone()[0]
        return {"chunks": chunks, "files": files, "images": images}

    def close(self):
        self.db.close()
