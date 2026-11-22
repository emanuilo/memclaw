"""Tests for MemoryIndex — particularly the embedding cache (spec #6)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from memclaw.config import ClaudeSettings, MemclawConfig, OpenAISettings
from memclaw.index import MemoryIndex


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai=OpenAISettings(api_key="test-key"),
        claude=ClaudeSettings(api_key="test-key"),
    )


@pytest.fixture
def idx(cfg: MemclawConfig) -> MemoryIndex:
    index = MemoryIndex(cfg)
    yield index
    index.close()


class TestEmbeddingCacheTable:
    def test_embedding_cache_table_exists(self, idx: MemoryIndex):
        """The embedding_cache table should be created on init."""
        row = idx.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
        ).fetchone()
        assert row is not None

    def test_embedding_cache_schema(self, idx: MemoryIndex):
        """Check the cache table has the expected columns."""
        cursor = idx.db.execute("PRAGMA table_info(embedding_cache)")
        cols = {row[1] for row in cursor.fetchall()}
        assert {"content_hash", "embedding", "model", "created_at"} <= cols


class TestEmbeddingCacheBehavior:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, idx: MemoryIndex):
        """On cache miss, the OpenAI API is called."""
        fake_emb = np.random.randn(1536).astype(np.float32)
        vectors = [fake_emb.tolist()]

        with patch.object(idx, "_openai") as mock_openai:
            mock_openai.embed = AsyncMock(return_value=vectors)
            idx._openai = mock_openai

            result = await idx.get_embeddings(["hello world"])
            assert len(result) == 1
            assert result[0].shape == (1536,)
            mock_openai.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, idx: MemoryIndex):
        """On cache hit (same hash + same model), the API is NOT called."""
        text = "hello world"
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        fake_emb = np.random.randn(1536).astype(np.float32)

        # Pre-populate the cache
        idx.db.execute(
            "INSERT INTO embedding_cache (content_hash, embedding, model) VALUES (?, ?, ?)",
            (content_hash, MemoryIndex.serialize_embedding(fake_emb), idx.config.memory.embedding_model),
        )
        idx.db.commit()

        with patch.object(idx, "_openai") as mock_openai:
            mock_openai.embed = AsyncMock()
            idx._openai = mock_openai

            result = await idx.get_embeddings([text])
            assert len(result) == 1
            np.testing.assert_array_almost_equal(result[0], fake_emb, decimal=5)
            # API should NOT have been called
            mock_openai.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_model(self, idx: MemoryIndex):
        """Cache entries with a different model are NOT used."""
        text = "hello world"
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        fake_emb = np.random.randn(1536).astype(np.float32)

        # Insert with a DIFFERENT model
        idx.db.execute(
            "INSERT INTO embedding_cache (content_hash, embedding, model) VALUES (?, ?, ?)",
            (content_hash, MemoryIndex.serialize_embedding(fake_emb), "text-embedding-OLD"),
        )
        idx.db.commit()

        new_emb = np.random.randn(1536).astype(np.float32)
        vectors = [new_emb.tolist()]

        with patch.object(idx, "_openai") as mock_openai:
            mock_openai.embed = AsyncMock(return_value=vectors)
            idx._openai = mock_openai

            await idx.get_embeddings([text])
            # API should have been called because model doesn't match
            mock_openai.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_cache_hits_and_misses(self, idx: MemoryIndex):
        """When some texts are cached and some aren't, only uncached hit the API."""
        cached_text = "cached"
        uncached_text = "uncached"

        cached_hash = hashlib.sha256(cached_text.encode()).hexdigest()
        cached_emb = np.random.randn(1536).astype(np.float32)
        idx.db.execute(
            "INSERT INTO embedding_cache (content_hash, embedding, model) VALUES (?, ?, ?)",
            (cached_hash, MemoryIndex.serialize_embedding(cached_emb), idx.config.memory.embedding_model),
        )
        idx.db.commit()

        uncached_emb = np.random.randn(1536).astype(np.float32)
        vectors = [uncached_emb.tolist()]

        with patch.object(idx, "_openai") as mock_openai:
            mock_openai.embed = AsyncMock(return_value=vectors)
            idx._openai = mock_openai

            result = await idx.get_embeddings([cached_text, uncached_text])
            assert len(result) == 2
            # First result should be the cached one
            np.testing.assert_array_almost_equal(result[0], cached_emb, decimal=5)
            # API called with only the uncached text
            call_args = mock_openai.embed.call_args
            assert call_args.args[0] == [uncached_text]

    @pytest.mark.asyncio
    async def test_empty_input(self, idx: MemoryIndex):
        """get_embeddings with empty list returns empty list without API calls."""
        result = await idx.get_embeddings([])
        assert result == []


class TestChunking:
    def test_basic_chunking(self, idx: MemoryIndex):
        text = "Hello world.\n\n---\n\nSecond paragraph."
        chunks = idx.chunk_text(text, "test.md")
        assert len(chunks) >= 1
        assert all(c.file_path == "test.md" for c in chunks)

    def test_empty_text(self, idx: MemoryIndex):
        chunks = idx.chunk_text("", "test.md")
        assert chunks == []


class TestEntryChunking:
    def test_each_entry_is_its_own_chunk(self, cfg: MemclawConfig, idx: MemoryIndex):
        from memclaw.memory import MemoryStore

        store = MemoryStore(cfg)
        store.save("First entry about apples.")
        path = store.save("Second entry about bananas.")
        chunks = idx.chunk_text(path.read_text(), str(path))
        bananas = [c for c in chunks if "bananas" in c.content]
        assert len(bananas) == 1
        assert "apples" not in bananas[0].content
        assert bananas[0].content.count("## ") == 1

    def test_long_entry_is_split_with_overlap_under_its_heading(self, idx: MemoryIndex):
        body = "\n".join(f"line {i} " + "word " * 20 for i in range(40))
        chunks = idx.chunk_text(f"## 10:00 - Note\n\n{body}\n\n---\n", "f.md")
        assert len(chunks) > 1
        assert all(c.content.startswith("## 10:00 - Note") for c in chunks)
        # Consecutive chunks share some text.
        assert chunks[0].content.splitlines()[-1] in chunks[1].content


class TestIndexScope:
    @pytest.mark.asyncio
    async def test_agents_md_and_hidden_dirs_are_not_indexed(self, cfg: MemclawConfig, idx: MemoryIndex):
        (cfg.memory_dir / ".cursor").mkdir()
        (cfg.memory_dir / ".cursor" / "notes.md").write_text("hidden")
        (cfg.memory_dir / "todos.md").write_text("- buy milk")
        await idx.sync()
        files = {Path(r[0]).name for r in idx.db.execute("SELECT file_path FROM file_meta")}
        assert files == {"MEMORY.md", "todos.md"}

    @pytest.mark.asyncio
    async def test_previously_indexed_agents_md_is_removed(self, cfg: MemclawConfig, idx: MemoryIndex):
        await idx.index_file(cfg.agent_file)  # what older versions did
        await idx.sync()
        rows = idx.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_path = ?", (str(cfg.agent_file),),
        ).fetchone()[0]
        assert rows == 0

    @pytest.mark.asyncio
    async def test_emptied_file_drops_its_chunks(self, cfg: MemclawConfig, idx: MemoryIndex):
        path = cfg.memory_dir / "todos.md"
        path.write_text("- buy milk")
        await idx.index_file(path)
        path.write_text("")
        await idx.index_file(path)
        assert idx.get_stats()["chunks"] == 0

    @pytest.mark.asyncio
    async def test_failed_embedding_keeps_previous_chunks(self, cfg: MemclawConfig, idx: MemoryIndex):
        path = cfg.memory_dir / "todos.md"
        path.write_text("- buy milk")
        await idx.index_file(path)
        path.write_text("- buy bread")
        idx.get_embeddings = AsyncMock(side_effect=RuntimeError("OpenAI down"))
        with pytest.raises(RuntimeError):
            await idx.index_file(path)
        contents = [r[0] for r in idx.db.execute("SELECT content FROM chunks")]
        assert contents == ["- buy milk"]

    @pytest.mark.asyncio
    async def test_sync_reindexes_a_file_with_an_older_mtime(self, cfg: MemclawConfig, idx: MemoryIndex):
        import os

        path = cfg.memory_dir / "todos.md"
        path.write_text("- buy milk")
        await idx.sync()
        path.write_text("- restored from backup")
        os.utime(path, (1_000_000, 1_000_000))
        assert await idx.sync()
        contents = [r[0] for r in idx.db.execute(
            "SELECT content FROM chunks WHERE file_path = ?", (str(path),))]
        assert contents == ["- restored from backup"]

    @pytest.mark.asyncio
    async def test_stats_count_images_from_every_platform(self, idx: MemoryIndex):
        await idx.store_platform_image("whatsapp", "wa-1", "a cat")
        await idx.store_platform_image("telegram", "tg-1", "a dog")
        assert idx.get_stats()["images"] == 2


class TestSearchQueries:
    """The queries HybridSearch runs through the index (it has no SQL of its own)."""

    @pytest.mark.asyncio
    async def test_chunks_and_their_embeddings(self, cfg: MemclawConfig, idx: MemoryIndex):
        assert idx.embeddings_version() == (0, 0)
        assert idx.chunk_embeddings() == []
        path = cfg.memory_subdir / "2026-01-01.md"
        path.write_text("## 10:00 - Note\n\nmy dog is called Max\n\n---\n")
        await idx.index_file(path)

        [(chunk_id, emb)] = idx.chunk_embeddings()
        assert idx.embeddings_version() == (1, chunk_id)
        assert emb.dtype == np.float32
        chunk = idx.get_chunk(chunk_id)
        assert (chunk.file_path, chunk.line_start, chunk.line_end) == (str(path), 0, 4)
        assert "my dog is called Max" in chunk.content
        assert idx.get_chunk(chunk_id + 1) is None

    @pytest.mark.asyncio
    async def test_keyword_matches(self, cfg: MemclawConfig, idx: MemoryIndex):
        path = cfg.memory_subdir / "2026-01-01.md"
        path.write_text("## 10:00 - Note\n\nmy dog is called Max\n\n---\n")
        await idx.index_file(path)
        [(chunk_id, rank)] = idx.keyword_matches('"max"', limit=None)
        assert idx.get_chunk(chunk_id) is not None and rank < 0
        assert idx.keyword_matches('"nothing"', limit=5) == []

    def test_a_malformed_keyword_query_finds_nothing(self, idx: MemoryIndex):
        assert idx.keyword_matches('"unterminated', limit=5) == []
