"""Retrieval end-to-end: real embeddings, real SQLite index, no LLM."""
from __future__ import annotations

from pathlib import Path

import pytest

from memclaw.agent import MemclawAgent
from memclaw.index import MemoryIndex
from memclaw.search import HybridSearch
from memclaw.store import MemoryStore
from memclaw.tools import ToolExecutor

from .conftest import days_ago, write_daily

# A small, realistic vault: (days ago, [(time, entry)]).
CORPUS: list[tuple[int, list[tuple[str, str]]]] = [
    (0, [
        ("09:10", "Plumber came to fix the kitchen sink. Invoice INV-48213, paid 120 EUR in cash."),
        ("13:45", "Lunch with Marko at Ambar. He recommended the book 'The Pragmatic Programmer'."),
    ]),
    (2, [
        ("08:30", "Ran 10k along the Danube in 52 minutes. Left knee felt a bit sore afterwards."),
        ("19:00", "Max (my beagle) has a vet appointment with Dr. Petrovic next Tuesday at 17:00."),
    ]),
    (5, [
        ("11:00", "Decided to migrate the Memclaw search index from FAISS to plain SQLite + numpy."),
        ("16:20", "Wi-Fi password for the cabin at Zlatibor is sunflower-42."),
    ]),
    (12, [
        ("10:00", "Sister Ana's birthday is on March 3rd. She likes dark chocolate and jazz vinyl."),
        ("18:30", "Booked flights to Lisbon for the conference, flight TP1237 departs 06:40."),
    ]),
    (40, [
        ("09:00", "Car registration renewal is due in November; the insurance policy number is AX-99-1203."),
        ("21:00", "Tried the new ramen place Ichiraku on Strahinjica Bana, the tonkotsu was excellent."),
    ]),
    (90, [
        ("12:00", "Started learning Spanish on Duolingo, goal is 15 minutes a day."),
    ]),
]

PERMANENT = """# Personal Memory

## People
- Ana: sister, lives in Novi Sad.
- Marko: former colleague, now at a fintech startup.

## Preferences
- Prefers oat milk in coffee.
"""

# (query, substring the top results must contain)
QUERIES: list[tuple[str, str]] = [
    ("What's the plumber's invoice number?", "INV-48213"),
    ("INV-48213", "INV-48213"),
    ("When is my dog's vet appointment?", "Petrovic"),
    ("cabin wifi password", "sunflower-42"),
    ("What does my sister like as a gift?", "dark chocolate"),
    ("Which flight am I taking to Lisbon?", "TP1237"),
    ("insurance policy number", "AX-99-1203"),
    ("good ramen restaurant", "Ichiraku"),
    ("How fast was my last run?", "52 minutes"),
    ("What did we decide about FAISS?", "SQLite"),
    ("Which book did Marko recommend?", "Pragmatic Programmer"),
    ("language learning goal", "Duolingo"),
    ("What milk do I take in my coffee?", "oat milk"),
]


async def _seed(vault) -> tuple[MemoryIndex, HybridSearch]:
    for ago, entries in CORPUS:
        write_daily(vault, days_ago(ago), entries)
    vault.memory_file.write_text(PERMANENT)
    index = MemoryIndex(vault)
    await index.sync()
    return index, HybridSearch(vault, index)


@pytest.mark.asyncio
async def test_recall_at_3(vault):
    """Every seeded fact is found in the top 3 for a natural question."""
    index, search = await _seed(vault)
    misses = []
    for query, expected in QUERIES:
        results = await search.search(query, limit=3)
        if not any(expected in r.content for r in results):
            misses.append((query, expected, [r.content[:60] for r in results]))
    index.close()
    assert not misses, "recall@3 misses:\n" + "\n".join(map(str, misses))


@pytest.mark.asyncio
async def test_keyword_channel_handles_natural_language(vault):
    """Punctuation in a question must not knock out keyword search, and an
    exact identifier must be found by the keyword channel."""
    index, search = await _seed(vault)
    hits = search._keyword_search("What's the plumber's invoice number?", limit=10)
    assert hits, "keyword search returned nothing for a natural-language question"
    hits = search._keyword_search("INV-48213", limit=10)
    assert hits, "keyword search returned nothing for an identifier with a hyphen"

    results = await search.search("INV-48213", limit=3)
    assert "INV-48213" in results[0].content
    assert results[0].match_type == "hybrid"
    index.close()


@pytest.mark.asyncio
async def test_keyword_best_match_scores_highest(vault):
    """The strongest BM25 match must get the highest keyword score."""
    write_daily(vault, days_ago(1), [
        ("10:00", "zebra zebra zebra zebra zebra crossing near the office"),
        ("11:00", "Saw a documentary; one scene briefly showed a zebra."),
    ])
    index = MemoryIndex(vault)
    await index.sync()
    search = HybridSearch(vault, index)
    hits = search._keyword_search("zebra", limit=10)
    by_id = dict(hits)
    rows = {
        cid: index.db.execute("SELECT content FROM chunks WHERE id=?", (cid,)).fetchone()[0]
        for cid in by_id
    }
    strong = next(cid for cid, text in rows.items() if "zebra zebra" in text)
    assert by_id[strong] == max(by_id.values())
    assert by_id[strong] > 0
    index.close()


@pytest.mark.asyncio
async def test_rewritten_file_is_searchable_with_warm_cache(vault):
    """Rewriting MEMORY.md with the same number of chunks (what consolidation
    does) must not leave search on a stale embedding matrix."""
    index, search = await _seed(vault)
    await search.search("anything", limit=3)  # warm the matrix cache

    vault.memory_file.write_text(PERMANENT.replace(
        "Prefers oat milk in coffee.", "Allergic to penicillin; carries an EpiPen.",
    ))
    await index.index_file(vault.memory_file)

    results = await search.search("Do I have any drug allergies?", limit=3)
    assert any("penicillin" in r.content for r in results)
    index.close()


@pytest.mark.asyncio
async def test_agents_md_is_not_a_memory(vault):
    """AGENTS.md is the agent's instructions (already in the system prompt),
    not something memory_search should return."""
    with open(vault.agent_file, "a") as f:
        f.write("\n- Always reply in Serbian, using the Latin alphabet.\n")
    index, search = await _seed(vault)
    results = await search.search("reply in Serbian Latin alphabet", limit=5)
    assert all(Path(r.file_path).name != "AGENTS.md" for r in results)
    index.close()


@pytest.mark.asyncio
async def test_emptied_file_drops_its_chunks(vault):
    index, search = await _seed(vault)
    path = vault.daily_file(days_ago(5))
    path.write_text("")
    await index.sync()
    results = await search.search("cabin wifi password", limit=5)
    assert not any("sunflower-42" in r.content for r in results)
    index.close()


@pytest.mark.asyncio
async def test_chunks_do_not_carry_previous_entry_heading(vault):
    """Each saved entry is its own chunk, without the heading or text of
    the entry before it."""
    store = MemoryStore(vault)
    store.save("First entry about apples.")
    path = store.save("Second entry about bananas.")
    index = MemoryIndex(vault)
    chunks = index.chunk_text(path.read_text(), str(path))
    bananas = [c for c in chunks if "bananas" in c.content]
    assert len(bananas) == 1
    assert "apples" not in bananas[0].content
    assert bananas[0].content.count("## ") == 1
    index.close()


@pytest.mark.asyncio
async def test_image_search_is_scoped_to_platform(vault):
    """An image saved on WhatsApp can't be sent through Telegram."""
    index = MemoryIndex(vault)
    search = HybridSearch(vault, index)
    await index.store_platform_image("whatsapp", "wa-media-1", "A whiteboard with sprint planning notes")
    await index.store_platform_image("telegram", "tg-file-1", "A whiteboard with sprint planning sticky notes")
    found: list[dict] = []
    tools = ToolExecutor(vault, MemoryStore(vault), index, search, found, platform="telegram")
    await tools.execute("image_search", {"query": "sprint planning whiteboard"})
    assert found and {img["platform"] for img in found} == {"telegram"}
    assert index.get_stats()["images"] == 2
    index.close()


@pytest.mark.asyncio
async def test_file_write_markdown_is_searchable(vault):
    index = MemoryIndex(vault)
    search = HybridSearch(vault, index)
    tools = ToolExecutor(vault, MemoryStore(vault), index, search, [])
    out = await tools.execute("file_write", {
        "file_path": "todos.md", "content": "# Todos\n\n- Renew passport before the Lisbon trip\n",
    })
    assert out.startswith("File written")
    results = await search.search("passport renewal", limit=3)
    assert any("passport" in r.content for r in results)
    index.close()


@pytest.mark.asyncio
async def test_file_write_cannot_clobber_internal_files(vault):
    index = MemoryIndex(vault)
    tools = ToolExecutor(vault, MemoryStore(vault), index, HybridSearch(vault, index), [])
    for target in ("memclaw.db", ".env", "sessions.json", "meta.json"):
        out = await tools.execute("file_write", {"file_path": target, "content": "oops"})
        assert out.startswith("Blocked"), (target, out)
    assert index.get_stats()["chunks"] == 0  # the DB still works
    index.close()


@pytest.mark.asyncio
async def test_context_survives_embedding_failure(vault, monkeypatch):
    """If OpenAI is unreachable, the per-turn context still builds (without
    memories) instead of failing the whole message."""
    agent = MemclawAgent(vault, backend=_NoBackend())
    monkeypatch.setattr(agent.index.openai, "api_key", "sk-invalid")
    context = await agent.build_context("hello there")
    assert context.startswith("<context>") and context.endswith("</context>")
    agent._close_sync_only()


class _NoBackend:
    name = "none"
    bills_per_token = False
