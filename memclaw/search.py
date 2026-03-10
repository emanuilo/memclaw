from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .config import MemclawConfig
from .index import MemoryIndex


@dataclass
class SearchResult:
    file_path: str
    line_start: int
    line_end: int
    content: str
    score: float
    match_type: str  # "hybrid", "vector", "keyword"


class HybridSearch:
    """Hybrid search combining vector cosine similarity and BM25 keyword matching.

    Inspired by OpenClaw's approach: weighted merge of vector and text scores
    with configurable weights (default 70 % vector / 30 % keyword).
    """

    def __init__(self, config: MemclawConfig, index: MemoryIndex):
        self.config = config
        self.index = index

    async def search_images(self, query: str, limit: int = 5) -> list[dict]:
        """Search stored Telegram images by semantic similarity."""
        await self.index.sync()
        query_embedding = await self.index.get_embedding(query)
        return self.index.search_telegram_images(query_embedding, limit=limit)

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        # Make sure the index is fresh before searching.
        await self.index.sync()

        query_embedding = await self.index.get_embedding(query)

        # Oversample: retrieve 3x candidates for MMR filtering
        internal_limit = limit * 3

        vector_hits = self._vector_search(query_embedding, limit=internal_limit)
        keyword_hits = self._keyword_search(query, limit=internal_limit)

        # Merge produces internal_limit candidates (not yet truncated)
        candidates = self._merge(vector_hits, keyword_hits, internal_limit)

        # Apply temporal decay (spec #4)
        candidates = self._apply_decay(candidates)

        # Apply MMR deduplication (spec #5) and return final top-k
        return self._apply_mmr(candidates, limit)

    # ------------------------------------------------------------------
    # Vector search (cosine similarity via numpy)
    # ------------------------------------------------------------------

    def _vector_search(
        self, query_embedding: np.ndarray, limit: int
    ) -> list[tuple[int, float]]:
        rows = self.index.db.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        ids = []
        embeddings = []
        for chunk_id, blob in rows:
            ids.append(chunk_id)
            embeddings.append(MemoryIndex.deserialize_embedding(blob))

        matrix = np.stack(embeddings)
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(query_embedding)
        similarities = (matrix @ query_embedding) / (norms * q_norm + 1e-8)

        top_idx = np.argsort(similarities)[::-1][:limit]
        return [(ids[i], float(similarities[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Keyword search (FTS5 / BM25)
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        try:
            rows = self.index.db.execute(
                """SELECT rowid, rank FROM chunks_fts
                   WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        except Exception:
            # FTS5 match can fail on certain query syntax
            return []

        if not rows:
            return []

        # FTS5 rank is negative (lower = better). Normalise to 0-1 positive.
        max_abs = max(abs(r[1]) for r in rows) or 1.0
        return [(row[0], 1.0 - abs(row[1]) / max_abs) for row in rows]

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(
        self,
        vector_hits: list[tuple[int, float]],
        keyword_hits: list[tuple[int, float]],
        limit: int,
    ) -> list[SearchResult]:
        vw = self.config.vector_weight
        tw = self.config.text_weight

        # Normalise vector scores to 0-1
        if vector_hits:
            max_v = max(s for _, s in vector_hits)
            min_v = min(s for _, s in vector_hits)
            range_v = max_v - min_v or 1.0
            v_scores = {cid: (s - min_v) / range_v for cid, s in vector_hits}
        else:
            v_scores = {}

        k_scores = dict(keyword_hits)

        all_ids = set(v_scores) | set(k_scores)
        combined: list[tuple[int, float, str]] = []
        for cid in all_ids:
            vs = v_scores.get(cid, 0.0)
            ks = k_scores.get(cid, 0.0)
            final = vw * vs + tw * ks
            if cid in v_scores and cid in k_scores:
                match_type = "hybrid"
            elif cid in v_scores:
                match_type = "vector"
            else:
                match_type = "keyword"
            combined.append((cid, final, match_type))

        combined.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for cid, score, match_type in combined[:limit]:
            row = self.index.db.execute(
                "SELECT file_path, line_start, line_end, content FROM chunks WHERE id = ?",
                (cid,),
            ).fetchone()
            if row:
                results.append(SearchResult(
                    file_path=row[0],
                    line_start=row[1],
                    line_end=row[2],
                    content=row[3],
                    score=score,
                    match_type=match_type,
                ))

        return results

    # ------------------------------------------------------------------
    # Temporal Decay (spec #4)
    # ------------------------------------------------------------------

    _DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")

    def _apply_decay(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply exponential time-decay to search scores.

        Chunks from files matching ``YYYY-MM-DD.md`` are decayed based on age.
        Evergreen files (MEMORY.md or anything not matching the date pattern)
        retain their full score.  If ``decay_half_life_days`` is 0, decay is
        disabled entirely.
        """
        half_life = self.config.decay_half_life_days
        if half_life <= 0:
            return results

        lam = math.log(2) / half_life
        today = date.today()

        for result in results:
            stem = Path(result.file_path).name
            m = self._DATE_RE.search(stem)
            if m is None:
                # Evergreen file (e.g. MEMORY.md) -- no decay
                continue
            file_date = date.fromisoformat(m.group(1))
            age_days = (today - file_date).days
            if age_days < 0:
                age_days = 0
            result.score *= math.exp(-lam * age_days)

        # Re-sort after decay
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # MMR Deduplication (spec #5)
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Word-level Jaccard similarity between two texts."""
        set_a = set(text_a.split())
        set_b = set(text_b.split())
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _apply_mmr(
        self, candidates: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance filtering.

        Greedily selects results that are both relevant (high score) and
        diverse (low similarity to already-selected results).
        """
        if not candidates:
            return []

        mmr_lambda = self.config.mmr_lambda
        selected: list[SearchResult] = []

        # Work with a mutable copy so we can remove picked items
        remaining = list(candidates)

        while remaining and len(selected) < limit:
            best_idx = -1
            best_mmr = -float("inf")

            for i, candidate in enumerate(remaining):
                relevance = candidate.score

                if selected:
                    max_sim = max(
                        self._jaccard_similarity(candidate.content, s.content)
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected
