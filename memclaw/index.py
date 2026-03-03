from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from .config import MemclawConfig


@dataclass
class Chunk:
    file_path: str
    line_start: int
    line_end: int
    content: str
    embedding: np.ndarray | None = None


class MemoryIndex:
    """SQLite-based index with FTS5 keyword search and numpy vector search.

    Embeddings are stored as BLOBs in SQLite and cosine similarity is computed
    in-memory with numpy — lightweight and dependency-free beyond numpy.
    """

    def __init__(self, config: MemclawConfig):
        self.config = config
        self._openai: AsyncOpenAI | None = None
        self._init_db()

    def _init_db(self):
        self.db = sqlite3.connect(str(self.config.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")

        self.db.execute("""
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

        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # Triggers keep FTS5 in sync with the chunks table.
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END
        """)
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS file_meta (
                file_path TEXT PRIMARY KEY,
                mtime REAL NOT NULL
            )
        """)

        # Telegram image registry — maps file_ids to descriptions for retrieval.
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS telegram_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                description TEXT NOT NULL,
                caption TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.db.commit()

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            self._openai = AsyncOpenAI(api_key=self.config.openai_api_key)
        return self._openai

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, file_path: str) -> list[Chunk]:
        """Split text into overlapping chunks by paragraphs and headings."""
        lines = text.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_start = 0
        current_heading = ""
        word_count = 0

        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_heading = line

            current_lines.append(line)
            word_count += len(line.split())

            is_boundary = (
                line.strip() == "---"
                or (line.startswith("#") and word_count > self.config.chunk_target_words // 2)
                or word_count >= self.config.chunk_target_words
            )

            if is_boundary and current_lines:
                content = "\n".join(current_lines).strip()
                if content and content != "---":
                    chunks.append(Chunk(
                        file_path=file_path,
                        line_start=current_start,
                        line_end=i,
                        content=content,
                    ))

                # Keep overlap for continuity
                overlap_words = 0
                overlap_start = len(current_lines)
                for j in range(len(current_lines) - 1, -1, -1):
                    overlap_words += len(current_lines[j].split())
                    if overlap_words >= self.config.chunk_overlap_words:
                        overlap_start = j
                        break

                current_lines = current_lines[overlap_start:]
                if current_heading and not any(l.startswith("#") for l in current_lines):
                    current_lines.insert(0, current_heading)
                current_start = max(0, i - len(current_lines) + 1)
                word_count = sum(len(l.split()) for l in current_lines)

        # Flush remaining content
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content and content != "---":
                chunks.append(Chunk(
                    file_path=file_path,
                    line_start=current_start,
                    line_end=len(lines) - 1,
                    content=content,
                ))

        return chunks

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        response = await self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

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

    async def index_file(self, file_path: Path):
        """Re-index a single file: chunk, embed, store."""
        content = file_path.read_text()
        if not content.strip():
            return

        mtime = file_path.stat().st_mtime
        path_str = str(file_path)

        # Remove stale chunks for this file
        self.db.execute("DELETE FROM chunks WHERE file_path = ?", (path_str,))

        chunks = self.chunk_text(content, path_str)
        if not chunks:
            return

        # Batch embed
        texts = [c.content for c in chunks]
        embeddings = await self.get_embeddings(texts)

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
        self.db.commit()

    async def sync(self) -> bool:
        """Re-index any memory files that have changed since last indexed.

        Returns True if anything was updated.
        """
        changed = False
        for md_file in sorted(self.config.memory_dir.glob("**/*.md")):
            mtime = md_file.stat().st_mtime
            row = self.db.execute(
                "SELECT mtime FROM file_meta WHERE file_path = ?",
                (str(md_file),),
            ).fetchone()
            if row is None or mtime > row[0]:
                await self.index_file(md_file)
                changed = True

        # Remove entries for deleted files
        stored = {r[0] for r in self.db.execute("SELECT file_path FROM file_meta").fetchall()}
        existing = {str(f) for f in self.config.memory_dir.glob("**/*.md")}
        for gone in stored - existing:
            self.db.execute("DELETE FROM chunks WHERE file_path = ?", (gone,))
            self.db.execute("DELETE FROM file_meta WHERE file_path = ?", (gone,))
            changed = True

        if changed:
            self.db.commit()
        return changed

    # ------------------------------------------------------------------
    # Telegram image registry
    # ------------------------------------------------------------------

    async def store_telegram_image(
        self,
        file_id: str,
        description: str,
        caption: str | None = None,
    ):
        """Store a Telegram image file_id with its AI-generated description."""
        embedding = await self.get_embedding(description)
        self.db.execute(
            "INSERT INTO telegram_images (file_id, description, caption, embedding) "
            "VALUES (?, ?, ?, ?)",
            (file_id, description, caption, self.serialize_embedding(embedding)),
        )
        self.db.commit()

    def search_telegram_images(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> list[dict]:
        """Vector search over stored Telegram images. Returns dicts with file_id."""
        rows = self.db.execute(
            "SELECT id, file_id, description, caption, embedding, created_at "
            "FROM telegram_images WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            stored_emb = self.deserialize_embedding(row[4])
            similarity = float(
                np.dot(query_embedding, stored_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb) + 1e-8)
            )
            results.append({
                "id": row[0],
                "file_id": row[1],
                "description": row[2],
                "caption": row[3],
                "created_at": row[5],
                "score": similarity,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = self.db.execute("SELECT COUNT(*) FROM file_meta").fetchone()[0]
        images = self.db.execute("SELECT COUNT(*) FROM telegram_images").fetchone()[0]
        return {"chunks": chunks, "files": files, "images": images}

    def close(self):
        self.db.close()
