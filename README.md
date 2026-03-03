<div align="center">

# Memclaw

**Your personal memory vault, powered by AI.**

Store your thoughts. Save your images. Ask anything, anytime.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

Memclaw is a lightweight, local-first personal memory assistant. It stores your thoughts, notes, and images as **plain Markdown files** and makes them instantly searchable through **hybrid vector + keyword search**.

Think of it as your second brain — one that actually remembers.

## Why Memclaw?

**Remember everything.** Drop in random thoughts, meeting notes, voice memos, or photos. Memclaw files them away and indexes them for instant retrieval — even months later.

**Find anything.** Ask natural language questions like *"What did I note about that restaurant?"* or *"What images did I save from vacation?"* Hybrid search combines semantic understanding with exact keyword matching so nothing slips through.

**Own your data.** Everything lives on your machine as plain Markdown files and a local SQLite database. No cloud storage. No subscriptions. No vendor lock-in. Your memories are yours — readable, editable, and version-controllable.

**Stay lightweight.** No Docker. No Postgres. No heavy infrastructure. Just Python, SQLite, and two API keys. Install in seconds, start remembering immediately.

## Quick Start

```bash
pip install memclaw
```

Set your API keys:

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
export OPENAI_API_KEY=your-openai-key
```

Launch:

```bash
memclaw
```

That's it. Start typing.

## How It Works

```
You ──► Memclaw Agent ──► Memory Tools ──► Markdown Files + SQLite Index
              │                                       │
              │         ◄── Hybrid Search ◄───────────┘
              │
              └──► Natural language response
```

Memclaw draws inspiration from [OpenClaw](https://github.com/openclaw/openclaw)'s memory architecture and is built with the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview).

### Storage Layer

All memories are plain Markdown — human-readable, editable, and git-friendly.

| File | Purpose |
|------|---------|
| `~/.memclaw/MEMORY.md` | Curated long-term facts and preferences |
| `~/.memclaw/memory/YYYY-MM-DD.md` | Timestamped daily entries |
| `~/.memclaw/memclaw.db` | SQLite index (vector embeddings + FTS5) |

### Search Layer

Every memory is chunked, embedded, and indexed in SQLite. Retrieval combines two signals:

- **Vector search** (70% weight) — cosine similarity via OpenAI embeddings finds semantically related memories even when wording differs
- **Keyword search** (30% weight) — BM25 via SQLite FTS5 catches exact tokens, names, and identifiers

### Agent Layer

Powered by Claude via the [Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview). The agent decides when to **store** vs **search** based on your intent, using three custom tools:

| Tool | What it does |
|------|-------------|
| `memory_save` | Writes a new entry to today's daily file or MEMORY.md |
| `memory_search` | Hybrid search across all indexed memories |
| `image_save` | Generates an AI description of an image and stores it |

## Usage

### Interactive Mode (default)

```bash
memclaw
```

Chat naturally:

```
> Just had coffee with Alex. She's moving to Berlin next month for a new role at Stripe.
✓ Memory saved

> Who is Alex?
Based on your memories, Alex is someone you had coffee with recently.
She's moving to Berlin for a new role at Stripe.

> /quit
```

### Direct Commands

These work without the Claude agent — only the OpenAI key is needed for embeddings.

```bash
# Save a quick note
memclaw save "Meeting at 3pm with the design team about the rebrand"

# Search your memories
memclaw search "design team meetings"

# Rebuild the search index
memclaw index

# Check your memory vault
memclaw status
```

### Saving Images

In interactive mode, tell the agent to save an image:

```
> Save the image at ~/photos/whiteboard.jpg — it's our sprint planning board
✓ Image saved: A whiteboard showing a sprint planning layout with colorful sticky notes...
```

The image is described by an AI vision model and the description is stored and indexed like any other memory.

## Configuration

Memclaw stores everything in `~/.memclaw/` by default. Override with `--memory-dir`:

```bash
memclaw --memory-dir ~/my-vault
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For interactive mode | Powers the Claude agent |
| `OPENAI_API_KEY` | Yes | Embeddings + image descriptions |

### Directory Structure

```
~/.memclaw/
├── MEMORY.md              # Permanent / curated memories
├── memclaw.db             # SQLite index (embeddings + FTS5)
└── memory/
    ├── 2025-06-15.md      # Daily notes
    ├── 2025-06-16.md
    └── ...
```

## Architecture

Inspired by [OpenClaw](https://github.com/openclaw/openclaw)'s approach to AI memory:

- **Markdown as source of truth** — all memories live as plain text you can read, edit, and version-control
- **SQLite for indexing** — zero-config database with FTS5 for keyword search and BLOBs for embeddings
- **NumPy for vectors** — cosine similarity computed in-memory, no native extensions required
- **Claude Agent SDK** — intelligent agent loop that autonomously decides how to handle your input
- **Chunking with overlap** — ~300-word chunks with 60-word overlap preserve context across boundaries

## Using as a Library

```python
import asyncio
from memclaw.config import MemclawConfig
from memclaw.store import MemoryStore
from memclaw.index import MemoryIndex
from memclaw.search import HybridSearch

async def main():
    config = MemclawConfig()
    store = MemoryStore(config)
    index = MemoryIndex(config)
    search = HybridSearch(config, index)

    # Save a memory
    path = store.save("The best pizza in town is at Mario's on 5th Ave")
    await index.index_file(path)

    # Search
    results = await search.search("pizza recommendations")
    for r in results:
        print(f"{r.score:.2f}: {r.content[:80]}")

    index.close()

asyncio.run(main())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenClaw](https://github.com/openclaw/openclaw) for the memory architecture inspiration
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) for the agent framework
- [SQLite FTS5](https://www.sqlite.org/fts5.html) for full-text search
