"""Shared fixtures for memclaw tests."""
from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from memclaw.config import MemclawConfig
from memclaw.index import MemoryIndex
from memclaw.search import HybridSearch, SearchResult
from memclaw.store import MemoryStore


@pytest.fixture
def tmp_config(tmp_path: Path) -> MemclawConfig:
    """MemclawConfig pointing at a fresh temp directory."""
    return MemclawConfig(
        memory_dir=tmp_path / "memclaw",
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )


@pytest.fixture
def store(tmp_config: MemclawConfig) -> MemoryStore:
    return MemoryStore(tmp_config)


@pytest.fixture
def index(tmp_config: MemclawConfig) -> MemoryIndex:
    idx = MemoryIndex(tmp_config)
    yield idx
    idx.close()


@pytest.fixture
def search(tmp_config: MemclawConfig, index: MemoryIndex) -> HybridSearch:
    return HybridSearch(tmp_config, index)


def make_fake_embedding(dim: int = 1536, seed: int = 0) -> np.ndarray:
    """Deterministic fake embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def make_search_result(
    file_path: str = "memory/2025-03-01.md",
    content: str = "test content",
    score: float = 0.8,
    match_type: str = "hybrid",
) -> SearchResult:
    return SearchResult(
        file_path=file_path,
        line_start=0,
        line_end=10,
        content=content,
        score=score,
        match_type=match_type,
    )
