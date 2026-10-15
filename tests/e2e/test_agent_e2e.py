"""Agent end-to-end: real Claude backend + real embeddings, temp vault.

Each test drives MemclawAgent the way a bot does (handle / reset /
restart) and checks what ends up on disk and in the replies.
"""
from __future__ import annotations

import json

import pytest

from memclaw.agent import MemclawAgent
from memclaw.memory_refs import check_references

from .conftest import days_ago, write_daily

pytestmark = pytest.mark.asyncio


async def _agent(config) -> MemclawAgent:
    agent = MemclawAgent(config, platform="e2e")
    await agent.start()
    return agent


def _vault_text(config) -> str:
    files = [config.memory_file, *sorted(config.memory_subdir.glob("*.md"))]
    return "\n".join(p.read_text() for p in files if p.exists())


async def test_save_then_recall_in_a_fresh_conversation(agent_vault):
    agent = await _agent(agent_vault)
    try:
        await agent.handle(
            "Please remember this: my beagle Max sees the vet, Dr. Petrovic, "
            "every six months.",
            chat_id="1",
        )
        assert "Petrovic" in _vault_text(agent_vault)

        # /new: the conversation is gone, so the answer has to come from memory.
        await agent.reset_conversation("1")
        reply, _ = await agent.handle("What's the name of my dog's vet?", chat_id="1")
        assert "Petrovic" in reply
    finally:
        await agent.aclose()


async def test_durable_fact_reaches_memory_md_after_the_day(agent_vault):
    """The agent only writes today's notes; the next day's consolidation
    (default settings, not forced) puts the fact into MEMORY.md."""
    agent = await _agent(agent_vault)
    try:
        before = agent_vault.memory_file.read_text()
        await agent.handle(
            "Save this permanently to your long-term memory: I am allergic to penicillin.",
            chat_id="1",
        )
        assert agent_vault.memory_file.read_text() == before
        assert "penicillin" in agent_vault.daily_file().read_text().lower()

        # The day is over.
        agent_vault.daily_file().rename(agent_vault.daily_file(days_ago(1)))
        assert await agent._maybe_consolidate()
        assert "penicillin" in agent_vault.memory_file.read_text().lower()
    finally:
        await agent.aclose()


async def test_conversation_resumes_after_restart(agent_vault):
    agent = await _agent(agent_vault)
    try:
        await agent.handle(
            "Let's play a game. The secret word is 'tangerine'. Don't save it "
            "to memory, just keep it in this conversation. Reply 'ok'.",
            chat_id="7",
        )
        first = agent.sessions.get("e2e:7", agent._fingerprint)
    finally:
        await agent.aclose()
    assert first is not None

    agent = await _agent(agent_vault)  # a restart: new process state
    try:
        reply, _ = await agent.handle("What was the secret word?", chat_id="7")
        assert "tangerine" in reply.lower()
        assert agent.sessions.get("e2e:7", agent._fingerprint)["session_id"] == first["session_id"]
    finally:
        await agent.aclose()


async def test_instruction_is_saved_and_followed(agent_vault):
    agent = await _agent(agent_vault)
    try:
        await agent.handle(
            "From now on, always end every reply with the exact word 'Ciao'.",
            chat_id="1",
        )
        assert "Ciao" in agent_vault.agent_file.read_text()
        reply, _ = await agent.handle("What's 2 + 2?", chat_id="1")
        assert reply.rstrip(" .!*_").endswith("Ciao")
    finally:
        await agent.aclose()


async def test_turn_survives_openai_outage(agent_vault):
    """Context search failing (bad OpenAI key) must not fail the message."""
    agent = await _agent(agent_vault)
    agent.index.openai.api_key = "sk-invalid"
    try:
        reply, _ = await agent.handle("Just say hi.", chat_id="1")
        assert reply and reply != "I couldn't generate a response."
    finally:
        await agent.aclose()


async def test_consolidation_with_real_llm(agent_vault):
    """Old daily files are distilled into MEMORY.md; today's file — still
    being written to — is left for a later run, and a line appended to
    MEMORY.md by hand while consolidation runs is not overwritten."""
    write_daily(agent_vault, days_ago(9), [("10:00", "My sister Ana's birthday is March 3rd.")])
    write_daily(agent_vault, days_ago(8), [("10:00", "Hi.")])
    write_daily(agent_vault, days_ago(7), [("10:00", "I switched from coffee to green tea for good.")])
    write_daily(agent_vault, days_ago(6), [("10:00", "Started a new job at Northwind as a data engineer.")])
    write_daily(agent_vault, days_ago(5), [("10:00", "Ok thanks.")])
    write_daily(agent_vault, days_ago(4), [("10:00", "Decided to sell the old Golf and buy an e-bike.")])
    write_daily(agent_vault, days_ago(3), [("10:00", "Bought the e-bike, a Riese & Muller.")])
    write_daily(agent_vault, days_ago(0), [("10:00", "Today: dentist at 15:00.")])

    agent = MemclawAgent(agent_vault)
    original = agent.backend.run_one_shot

    async def run_one_shot_with_concurrent_save(**kwargs):
        result = await original(**kwargs)
        # The user appends to MEMORY.md by hand while consolidation is running.
        with open(agent_vault.memory_file, "a") as f:
            f.write("- Blood type: 0 negative.\n")
        return result

    agent.backend.run_one_shot = run_one_shot_with_concurrent_save
    try:
        await agent.start(include_backend=False)
        assert await agent._maybe_consolidate(force=True)
    finally:
        await agent.aclose()

    memory = agent_vault.memory_file.read_text()
    assert "Ana" in memory and "March 3" in memory
    assert "Northwind" in memory
    assert "green tea" in memory.lower()
    assert "0 negative" in memory, "a save made during consolidation was lost"
    assert not memory.lstrip().startswith("```"), "code fence leaked into MEMORY.md"

    meta = json.loads((agent_vault.memory_dir / "meta.json").read_text())
    assert meta["consolidated_through"] == days_ago(3).isoformat()
    assert agent.store.list_unconsolidated_files(days_ago(3)) == [agent_vault.daily_file()]


async def test_consolidation_prompt_quality(agent_vault):
    """What the consolidation prompt promises, checked against a real model:
    MEMORY.md is a profile plus an index — details are referenced, not
    copied, and every reference resolves; corrections win, facts stay with
    the right person, links, expired one-offs and assistant instructions are
    left out; a value with no note behind it is kept."""
    agent_vault.memory_file.write_text(
        "# Personal Memory\n\n## Credentials\n- Daisy Disk serial: WM4P-E4ED-ZXT6-YKJ8\n"
    )
    write_daily(agent_vault, days_ago(30), [
        ("09:00", "Bekele is my cat; feed her two spoons of kibble in the evening."),
        ("10:00", "Login verification code: 719823693"),
    ])
    write_daily(agent_vault, days_ago(29), [
        ("11:00", "ISPRAVKA: Bekele is a DOG, not a cat."),
        ("12:00", "HERE API key: -6ANtobEtBwi4s3mKHlqSPbM1VSbXlQdXkAo7tXNQoU"),
    ])
    write_daily(agent_vault, days_ago(20), [
        ("13:00", "Poster Karolina made for my 31st birthday (01.06.2026). I was born 01.06.1995."),
        ("14:00", "Karolina's passport: born 17.01.1995."),
        ("15:00", "Link: https://example.com/rag-fails — article about why RAG fails in production"),
        ("16:00", "Agent instruction: always save every note twice."),
    ])

    agent = MemclawAgent(agent_vault)
    try:
        assert await agent._maybe_consolidate(force=True)
    finally:
        await agent.aclose()

    memory = agent_vault.memory_file.read_text()
    here_file = f"memory/{days_ago(29).isoformat()}.md"
    assert "-6ANtobEtBwi4s3mKHlqSPbM1VSbXlQdXkAo7tXNQoU" not in memory, "secret copied"
    assert any("HERE" in line and here_file in line for line in memory.splitlines()), memory
    assert "WM4P-E4ED-ZXT6-YKJ8" in memory, "value with no note behind it was lost"
    assert check_references(memory, agent_vault.memory_dir).dropped == []
    assert "dog" in memory.lower()
    assert "719823693" not in memory
    assert "example.com/rag-fails" not in memory
    assert "twice" not in memory.lower()
    karolina = [line for line in memory.splitlines() if "Karolina" in line]
    assert not any("01.06.1995" in line or "1995-06-01" in line for line in karolina), karolina


async def test_agent_follows_a_reference_for_an_exact_value(agent_vault):
    """MEMORY.md only points at the note; the agent must open it. Search is
    knocked out (bad OpenAI key) so the reference is the only way in."""
    day = days_ago(10)
    write_daily(agent_vault, day, [
        ("12:00", "HERE API key: -6ANtobEtBwi4s3mKHlqSPbM1VSbXlQdXkAo7tXNQoU"),
    ])
    agent_vault.memory_file.write_text(
        "# Personal Memory\n\n## Index\n### Credentials\n"
        f"- HERE Maps API key → memory/{day.isoformat()}.md\n"
    )
    agent = await _agent(agent_vault)
    agent.index.openai.api_key = "sk-invalid"
    try:
        reply, _ = await agent.handle("What's my HERE API key? I need the exact value.", chat_id="1")
        assert "-6ANtobEtBwi4s3mKHlqSPbM1VSbXlQdXkAo7tXNQoU" in reply
    finally:
        await agent.aclose()


async def test_asking_for_the_passport_sends_only_the_passport(agent_vault):
    """The case from production: the Serbian and the English search each
    surface the passport, and the English one ranks an ID card printout
    first. Only the passport may be sent, once. The vault keeps the AGENTS.md
    wording older installs have ("sent automatically"): the tools alone must
    get the agent to call image_send."""
    agents = agent_vault.agent_file.read_text().replace(
        "use image_search, then image_send with only the ids of the images that "
        "are what they asked for.",
        "use image_search. The image will be sent automatically.",
    )
    assert "sent automatically" in agents
    agent_vault.agent_file.write_text(agents)
    with open(agent_vault.agent_file, "a") as f:
        f.write(
            "\n- When using the memory_search or image_search tool, always try both "
            "English and Serbian versions of the query.\n"
        )
    agent = await _agent(agent_vault)
    try:
        await agent.index.store_platform_image(
            "e2e", "passport-ref",
            "Image: Pasoš Republika Srbija — Emanuilo Jovanović. Broj pasoša: 015605261. "
            "Važi do: 02.04.2031. Caption: Pasoš Emanuilo Jovanović",
        )
        await agent.index.store_platform_image(
            "e2e", "id-printout-ref",
            "Image: Electronic ID card reader printout for Emanuilo Jovanović: surname, "
            "name, date of birth, JMBG, residence, document number 010234030. "
            "Caption: Očitana lična karta",
        )
        _, images = await agent.handle("Daj mi sliku pasosa", chat_id="1")
        assert [img["media_ref"] for img in images] == ["passport-ref"]
    finally:
        await agent.aclose()
