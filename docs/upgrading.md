# Upgrading

## Library: grouped settings (next release)

Nothing changes for the `memclaw` command or your `.env`. In the library, `MemclawConfig`'s settings moved into groups: `MemclawConfig(openai_api_key="k")` is now `MemclawConfig(openai=OpenAISettings(api_key="k"))`, and `config.slack_bot_token` is `config.slack.bot_token`, `config.mmr_lambda` is `config.memory.mmr_lambda`, and so on. `memory_dir`, `agent_backend` and `platform` stay where they were. See [Using Memclaw as a Library](library.md).

The store, index and search moved into the `memclaw.memory` package: `from memclaw.store import MemoryStore`, `from memclaw.index import MemoryIndex` and `from memclaw.search import HybridSearch, SearchResult` are now `from memclaw.memory import MemoryStore, MemoryIndex, HybridSearch, SearchResult`. They work as before.

## From 2.x to 3.0

```bash
pip install --upgrade memclaw            # or: pipx upgrade memclaw
pip install --upgrade "memclaw[cursor]"  # with the Cursor backend
```

No config changes are needed: your `.env`, memories and index carry over as they are. What's different:

- `MEMORY.md` becomes a short profile of you plus an index pointing to the daily notes that hold the details (credentials, documents, instructions). The first consolidation after upgrading restructures it; the previous version is kept in `~/.memclaw/.history/`.
- Consolidation runs after each day with notes, on Claude Opus 5.5 at medium effort (`CLAUDE_CONSOLIDATION_MODEL` / `CLAUDE_CONSOLIDATION_EFFORT` to change it). Chats default to `claude-sonnet-5-5`.
- Only consolidation writes `MEMORY.md`. `memclaw save --permanent` is gone; in the library, `MemoryStore.save()` no longer takes `permanent`, and `MemclawConfig` no longer takes `conversation_history_limit` / `conversation_history_window_minutes`.
- Each chat resumes its conversation after a restart (the first run after upgrading starts fresh). Memclaw no longer loads your `~/.claude` settings, plugins or MCP servers.

The full list of changes is in the [v3.0.0 release notes](https://github.com/emanuilo/memclaw/releases/tag/v3.0.0).
