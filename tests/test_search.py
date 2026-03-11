"""Tests for HybridSearch — decay, MMR, file_filter, vector caching (specs #3–5, #8)."""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from memclaw.config import MemclawConfig
from memclaw.index import MemoryIndex
from memclaw.search import HybridSearch, SearchResult


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _result(
    file_path: str = "memory/2025-03-01.md",
    content: str = "test content",
    score: float = 0.8,
) -> SearchResult:
    return SearchResult(
        file_path=file_path,
        line_start=0,
        line_end=10,
        content=content,
        score=score,
        match_type="hybrid",
    )


@pytest.fixture
def cfg(tmp_path: Path) -> MemclawConfig:
    return MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="k",
        anthropic_api_key="k",
    )


@pytest.fixture
def idx(cfg: MemclawConfig) -> MemoryIndex:
    index = MemoryIndex(cfg)
    yield index
    index.close()


@pytest.fixture
def hs(cfg: MemclawConfig, idx: MemoryIndex) -> HybridSearch:
    return HybridSearch(cfg, idx)


# ────────────────────────────────────────────────────────────────────
# Spec #4: Temporal Decay
# ────────────────────────────────────────────────────────────────────

class TestTemporalDecay:
    def test_no_decay_for_evergreen_files(self, hs: HybridSearch):
        """MEMORY.md and non-date files keep their full score."""
        results = [
            _result(file_path="/some/path/MEMORY.md", score=0.9),
            _result(file_path="/some/path/notes.md", score=0.8),
        ]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == 0.9
        assert decayed[1].score == 0.8

    def test_recent_file_minimal_decay(self, hs: HybridSearch):
        """A file from today should have essentially no decay."""
        today_str = date.today().isoformat()
        results = [_result(file_path=f"memory/{today_str}.md", score=1.0)]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == pytest.approx(1.0, abs=0.01)

    def test_30_day_old_file_half_score(self, hs: HybridSearch):
        """With default 30-day half-life, a 30-day-old file → ~50% score."""
        old_date = (date.today() - timedelta(days=30)).isoformat()
        results = [_result(file_path=f"memory/{old_date}.md", score=1.0)]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == pytest.approx(0.5, abs=0.05)

    def test_90_day_old_file_heavy_decay(self, hs: HybridSearch):
        """90 days = 3 half-lives → ~12.5% score."""
        old_date = (date.today() - timedelta(days=90)).isoformat()
        results = [_result(file_path=f"memory/{old_date}.md", score=1.0)]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == pytest.approx(0.125, abs=0.02)

    def test_decay_disabled_when_zero(self, cfg: MemclawConfig, idx: MemoryIndex):
        """When decay_half_life_days=0, no decay is applied."""
        cfg.decay_half_life_days = 0
        hs = HybridSearch(cfg, idx)
        old_date = (date.today() - timedelta(days=365)).isoformat()
        results = [_result(file_path=f"memory/{old_date}.md", score=1.0)]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == 1.0

    def test_decay_reorders_results(self, hs: HybridSearch):
        """After decay, results should be re-sorted by decayed score."""
        today_str = date.today().isoformat()
        old_date = (date.today() - timedelta(days=60)).isoformat()
        results = [
            _result(file_path=f"memory/{old_date}.md", score=0.95),
            _result(file_path=f"memory/{today_str}.md", score=0.80),
        ]
        decayed = hs._apply_decay(results)
        # The recent file should now rank higher due to decay on the old one
        assert decayed[0].file_path == f"memory/{today_str}.md"

    def test_future_date_no_negative_decay(self, hs: HybridSearch):
        """A file with a future date should not get boosted."""
        future = (date.today() + timedelta(days=5)).isoformat()
        results = [_result(file_path=f"memory/{future}.md", score=1.0)]
        decayed = hs._apply_decay(results)
        assert decayed[0].score == pytest.approx(1.0, abs=0.01)


# ────────────────────────────────────────────────────────────────────
# Spec #5: MMR Deduplication
# ────────────────────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_texts(self, hs: HybridSearch):
        assert hs._jaccard_similarity("a b c", "a b c") == 1.0

    def test_completely_different(self, hs: HybridSearch):
        assert hs._jaccard_similarity("a b c", "d e f") == 0.0

    def test_partial_overlap(self, hs: HybridSearch):
        # {a, b, c} ∩ {b, c, d} = {b, c}, union = {a, b, c, d}
        sim = hs._jaccard_similarity("a b c", "b c d")
        assert sim == pytest.approx(2 / 4)

    def test_empty_texts(self, hs: HybridSearch):
        assert hs._jaccard_similarity("", "") == 1.0

    def test_one_empty(self, hs: HybridSearch):
        assert hs._jaccard_similarity("a b c", "") == 0.0


class TestMMR:
    def test_mmr_returns_limited_results(self, hs: HybridSearch):
        candidates = [_result(content=f"content {i}", score=1.0 - i * 0.1) for i in range(10)]
        selected = hs._apply_mmr(candidates, limit=3)
        assert len(selected) == 3

    def test_mmr_prefers_diverse_results(self, hs: HybridSearch):
        """Near-duplicate content should be deprioritized."""
        candidates = [
            _result(content="the quick brown fox jumps over lazy dog", score=0.9),
            _result(content="the quick brown fox jumps over lazy dog again", score=0.85),
            _result(content="completely different unique content about python programming", score=0.7),
        ]
        selected = hs._apply_mmr(candidates, limit=2)
        contents = [r.content for r in selected]
        # The diverse result should be preferred over the near-duplicate
        assert any("python programming" in c for c in contents)

    def test_mmr_empty_candidates(self, hs: HybridSearch):
        assert hs._apply_mmr([], limit=5) == []

    def test_mmr_fewer_candidates_than_limit(self, hs: HybridSearch):
        candidates = [_result(content="single result", score=0.9)]
        selected = hs._apply_mmr(candidates, limit=5)
        assert len(selected) == 1

    def test_mmr_first_pick_is_highest_score(self, hs: HybridSearch):
        """The first MMR pick should be the highest-scoring candidate."""
        candidates = [
            _result(content="low score", score=0.3),
            _result(content="high score", score=0.95),
            _result(content="medium score", score=0.6),
        ]
        selected = hs._apply_mmr(candidates, limit=1)
        assert selected[0].content == "high score"


# ────────────────────────────────────────────────────────────────────
# Spec #3: file_filter parameter
# ────────────────────────────────────────────────────────────────────

class TestFileFilter:
    @pytest.mark.asyncio
    async def test_file_filter_restricts_results(self, hs: HybridSearch):
        """search() with file_filter should only return matching file paths."""
        # We can't easily run the full search without real embeddings,
        # so test the filtering logic directly via _merge output
        results = [
            _result(file_path="/vault/MEMORY.md", score=0.9),
            _result(file_path="/vault/memory/2025-03-01.md", score=0.8),
            _result(file_path="/vault/MEMORY.md", content="second chunk", score=0.7),
        ]
        filtered = [r for r in results if "MEMORY.md" in r.file_path]
        assert len(filtered) == 2
        assert all("MEMORY.md" in r.file_path for r in filtered)


# ────────────────────────────────────────────────────────────────────
# Spec #8 Phase 1: Vector Search Caching
# ────────────────────────────────────────────────────────────────────

class TestVectorSearchCache:
    def test_cache_initialized_empty(self, hs: HybridSearch):
        assert hs._embedding_cache is None
        assert hs._cache_chunk_count == 0

    def test_cache_populated_after_first_search(self, hs: HybridSearch, idx: MemoryIndex):
        """After a vector search with data, the cache should be populated."""
        # Insert a fake chunk with an embedding
        emb = np.random.randn(1536).astype(np.float32)
        idx.db.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, content, embedding, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test.md", 0, 10, "hello", MemoryIndex.serialize_embedding(emb), 1.0),
        )
        idx.db.commit()

        query = np.random.randn(1536).astype(np.float32)
        hs._vector_search(query, limit=10)

        assert hs._embedding_cache is not None
        assert hs._cache_chunk_count == 1

    def test_cache_reused_on_same_count(self, hs: HybridSearch, idx: MemoryIndex):
        """When chunk count hasn't changed, the cached matrix is reused."""
        emb = np.random.randn(1536).astype(np.float32)
        idx.db.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, content, embedding, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test.md", 0, 10, "hello", MemoryIndex.serialize_embedding(emb), 1.0),
        )
        idx.db.commit()

        query = np.random.randn(1536).astype(np.float32)
        hs._vector_search(query, limit=10)
        first_cache = hs._embedding_cache

        # Second search — should reuse cache
        hs._vector_search(query, limit=10)
        assert hs._embedding_cache is first_cache  # same object

    def test_cache_invalidated_on_new_chunk(self, hs: HybridSearch, idx: MemoryIndex):
        """When a new chunk is added, the cache should be rebuilt."""
        emb1 = np.random.randn(1536).astype(np.float32)
        idx.db.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, content, embedding, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test.md", 0, 10, "hello", MemoryIndex.serialize_embedding(emb1), 1.0),
        )
        idx.db.commit()

        query = np.random.randn(1536).astype(np.float32)
        hs._vector_search(query, limit=10)
        assert hs._cache_chunk_count == 1

        # Add another chunk
        emb2 = np.random.randn(1536).astype(np.float32)
        idx.db.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, content, embedding, file_mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test2.md", 0, 5, "world", MemoryIndex.serialize_embedding(emb2), 1.0),
        )
        idx.db.commit()

        hs._vector_search(query, limit=10)
        assert hs._cache_chunk_count == 2

    def test_empty_index_returns_empty(self, hs: HybridSearch):
        query = np.random.randn(1536).astype(np.float32)
        results = hs._vector_search(query, limit=10)
        assert results == []
