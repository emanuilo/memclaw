# Upgrading

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
