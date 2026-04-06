<div align="center">

<img src="docs/assets/memclaw-logo.png" alt="Memclaw" width="500">

**Your personal memory vault, powered by AI.**

Store your thoughts. Save your images and links. Ask anything, anytime.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Anthropic API](https://img.shields.io/badge/Anthropic-API-blueviolet.svg)](https://docs.anthropic.com/)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-26A5E4.svg)](https://core.telegram.org/bots)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-Bot-25D366.svg)](https://developers.facebook.com/docs/whatsapp/cloud-api)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

Memclaw is a lightweight, local-first personal memory assistant. It stores your thoughts, notes, images, and links as **plain Markdown files** and makes them instantly searchable through **hybrid vector + keyword search**.

Think of it as your second brain — one that actually remembers.

## Why Memclaw?

[OpenClaw](https://github.com/openclaw/openclaw) is great — but it connects to dozens of tools, reads your filesystem, runs shell commands, and does a hundred things you didn't ask for. That makes it slow, expensive, and a security surface you have to think about every time you use it.

Memclaw takes one slice of what OpenClaw does — **memory** — and does it really well, without the rest.

**Total recall.** Save a thought today, retrieve it six months later with a vague description. *"What was that restaurant Alex told me about?"* — Memclaw finds it. Hybrid vector + keyword search means you don't need to remember exact words.

**Visual memory.** Send an image and Memclaw generates an AI caption, indexes it, and stores it. Later, just ask *"that food recipe photo"* or *"the whiteboard from last sprint"* and it comes right back.

**Link memory.** Drop a link and Memclaw fetches the page, summarizes it, and indexes the content. Months later, ask *"that article about distributed databases"* and it surfaces the link with context — no bookmarking app needed.

**Sandboxed by design.** Memclaw only touches `~/.memclaw/`. No filesystem access, no shell commands, no path traversal. You don't need to trust it with your whole computer — it can't see it.

**Lightweight and cheap.** No Docker. No Postgres. No sprawling tool graph burning tokens. Just Python, SQLite, and two API keys. Fast responses, minimal cost.

## Quick Start

```bash
pip install memclaw
memclaw
```

On first run, Memclaw will prompt you for your API keys and save them to `~/.memclaw/.env`. You can update them anytime with `memclaw configure`.

## Messaging Platforms

The main way to use Memclaw. Just talk to it naturally — no commands needed. Send text, photos, voice messages, or links. The agent figures out what to do: store it, search your memories, retrieve images, or just chat.

Both platforms share the same agent, memories, and search index — your data is unified regardless of how you interact.

### Telegram Bot

The Telegram bot shows a **typing indicator** while processing so you know it's working on your request.

#### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Get your Telegram user ID (e.g. via [@userinfobot](https://t.me/userinfobot)).
3. Start the bot — on first run you'll be prompted for all keys:

```bash
memclaw bot
```

### WhatsApp Bot

The WhatsApp bot runs as an HTTP webhook server that receives messages from the WhatsApp Cloud API.

#### Setup

1. Create a [Meta Business App](https://developers.facebook.com/apps/) and enable the WhatsApp product.
2. In the WhatsApp section, set up a phone number and note the **Phone Number ID**.
3. Generate a permanent **access token** (or use the temporary one for testing).
4. Choose a **verify token** (any secret string you pick).
5. Start the webhook server:

```bash
memclaw whatsapp --port 8080
```

6. Configure the webhook URL in your Meta app dashboard:
   - URL: `https://your-server:8080/webhook`
   - Verify token: the value you set as `WHATSAPP_VERIFY_TOKEN`
   - Subscribe to: `messages`

> You'll need a publicly accessible HTTPS URL. Use a reverse proxy (nginx, Caddy) or a tunnel (ngrok, Cloudflare Tunnel) during development.

To update keys later: `memclaw configure`.

### What it handles

| Message type | What happens |
|-------------|-------------|
| **Text** | Agent decides: store as memory, search existing memories, or both. Links are extracted, fetched, and summarized automatically. |
| **Photo** | AI-described via vision model, stored and indexed. Agent acknowledges and responds. Saved for later retrieval. |
| **Voice** | Transcribed via Whisper, stored as text. Agent responds to the content. Links extracted. |

### Examples

```
> Just had coffee with Alex. She's moving to Berlin for a role at Stripe.
Got it! I've saved that Alex is moving to Berlin for a new role at Stripe.

> Who is Alex?
Based on your memories, Alex is someone you had coffee with recently.
She's moving to Berlin for a new role at Stripe.

> Show me the whiteboard photo from last week
[sends the matching photo]
Here's the sprint planning whiteboard you saved last week.
```

## How It Works

```mermaid
flowchart LR
    TG[Telegram] -->|text / images / links| Agent[Memclaw Agent]
    WA[WhatsApp] -->|text / images / links| Agent

    subgraph sandbox ["~/.memclaw/"]
        Agent -->|save| Tools1["memory_save<br>image_save<br>file_write"]
        Tools1 --> Files["MEMORY.md<br>daily/*.md<br>AGENTS.md"]
        Files --> DB[("SQLite<br>FTS5 + Vectors")]
        DB --> Tools2["memory_search<br>image_search"]
        Tools2 -->|results| Agent
    end

    Agent -->|response + images| TG
    Agent -->|response + images| WA
```

Memclaw draws inspiration from [OpenClaw](https://github.com/openclaw/openclaw)'s memory architecture and uses the [Anthropic API](https://docs.anthropic.com/) directly with a lightweight agentic loop.

### Storage Layer

All memories are plain Markdown — human-readable, editable, and git-friendly.

| File | Purpose |
|------|---------|
| `~/.memclaw/MEMORY.md` | Curated long-term facts and preferences |
| `~/.memclaw/AGENTS.md` | Customizable agent instructions and user preferences |
| `~/.memclaw/memory/YYYY-MM-DD.md` | Timestamped daily entries |
| `~/.memclaw/memclaw.db` | SQLite index (vector embeddings + FTS5) |

### Search Layer

Every memory is chunked, embedded, and indexed in SQLite. Retrieval combines two signals:

- **Vector search** (70% weight) — cosine similarity via OpenAI embeddings finds semantically related memories even when wording differs
- **Keyword search** (30% weight) — BM25 via SQLite FTS5 catches exact tokens, names, and identifiers
- **Temporal decay** — recent memories score higher (30-day half-life), MEMORY.md entries are evergreen
- **MMR deduplication** — removes near-duplicate results to keep search diverse

### Agent Layer

Powered by Claude via the [Anthropic API](https://docs.anthropic.com/) with a hand-rolled agentic loop. The agent maintains a rolling 10-message-pair conversation history and decides when to **store** vs **search** based on your intent.

| Tool | What it does |
|------|-------------|
| `memory_save` | Writes a new entry to today's daily file or MEMORY.md |
| `memory_search` | Hybrid search across all indexed memories |
| `image_save` | Generates an AI description of an image and stores it |
| `image_search` | Retrieves previously stored images by description |
| `file_write` / `file_read` | Sandboxed file operations within `~/.memclaw/` |
| `update_instructions` | Appends user preferences to AGENTS.md |

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

# Consolidate daily files into MEMORY.md
memclaw consolidate

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

On first run, `memclaw` or `memclaw bot` will launch an interactive setup wizard that saves your keys to `~/.memclaw/.env`. Run `memclaw configure` anytime to update them.

You can also set keys via environment variables or a `.env` in the current directory — these take the usual precedence over the saved config.

```bash
memclaw --memory-dir ~/my-vault   # override storage location
```

### API Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Embeddings + image descriptions |
| `ANTHROPIC_API_KEY` | Yes | Powers the Claude agent |
| `TELEGRAM_BOT_TOKEN` | For Telegram bot | Your Telegram bot token |
| `ALLOWED_USER_IDS` | For Telegram bot | Comma-separated Telegram user IDs |
| `WHATSAPP_PHONE_NUMBER_ID` | For WhatsApp bot | Your WhatsApp Business phone number ID |
| `WHATSAPP_ACCESS_TOKEN` | For WhatsApp bot | WhatsApp Cloud API access token |
| `WHATSAPP_VERIFY_TOKEN` | For WhatsApp bot | Secret string for webhook verification |
| `WHATSAPP_ALLOWED_NUMBERS` | For WhatsApp bot | Comma-separated phone numbers (e.g. +1234567890) |

### Directory Structure

```
~/.memclaw/
├── MEMORY.md              # Permanent / curated memories
├── AGENTS.md              # Agent instructions + user preferences
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
- **Auto-consolidation** — daily files are periodically distilled into MEMORY.md
- **Filesystem guardrail** — SDK-level callback blocks all file access outside `~/.memclaw/`
- **Embedding cache** — SHA-256 content hashing skips redundant API calls

For a deep dive into how memory storage, search, consolidation, and context injection work, see the [Memory Architecture](docs/memory-architecture.md) doc.

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
- [Anthropic API](https://docs.anthropic.com/) for the agent framework
- [SQLite FTS5](https://www.sqlite.org/fts5.html) for full-text search
