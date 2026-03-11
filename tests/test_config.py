"""Tests for new MemclawConfig fields (specs #1, #2, #4, #5)."""
from pathlib import Path

from memclaw.config import MemclawConfig


def test_default_conversation_history_limit(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.conversation_history_limit == 10


def test_default_consolidation_threshold(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.consolidation_threshold == 7


def test_default_decay_half_life_days(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.decay_half_life_days == 30


def test_default_mmr_lambda(tmp_path: Path):
    cfg = MemclawConfig(memory_dir=tmp_path / "m", openai_api_key="k", anthropic_api_key="k")
    assert cfg.mmr_lambda == 0.7


def test_custom_values(tmp_path: Path):
    cfg = MemclawConfig(
        memory_dir=tmp_path / "m",
        openai_api_key="k",
        anthropic_api_key="k",
        conversation_history_limit=5,
        consolidation_threshold=3,
        decay_half_life_days=60,
        mmr_lambda=0.5,
    )
    assert cfg.conversation_history_limit == 5
    assert cfg.consolidation_threshold == 3
    assert cfg.decay_half_life_days == 60
    assert cfg.mmr_lambda == 0.5
