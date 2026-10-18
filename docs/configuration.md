# Configuration

On first run, `memclaw` launches a setup wizard: it asks which front-end you want (Telegram, WhatsApp, Slack or Terminal) and which agent backend, prompts only for the keys those need, and saves them to `~/.memclaw/.env`. Run `memclaw configure` anytime to switch platform or backend, or to update keys.

You can also set keys via environment variables or a `.env` in the current directory — these take the usual precedence over the saved config.

```bash
memclaw --memory-dir ~/my-vault   # override storage location
```

## Agent backend

Memclaw runs your messages through a pluggable agent SDK. Only the chosen backend's keys are saved.

### Claude (default)

Via [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/). The wizard asks for one of two auth modes:

- **Claude subscription** (Pro / Max / Team) — generate a long-lived token with `claude setup-token`; Memclaw saves it as `CLAUDE_CODE_OAUTH_TOKEN`. Requests bill against your plan, and the per-message cost line is omitted from logs.
- **Anthropic API key** (pay-as-you-go) — set `ANTHROPIC_API_KEY` to an `sk-ant-…` key. Requests bill per token, and the cost is logged with each turn.

Only one of the two is set at a time.

The wizard then lets you pick a model from the list Anthropic offers your credential and, if the model supports it, an effort level. Pressing Enter keeps your current choice. The default is `claude-sonnet-5-5` at `medium` effort. Memory consolidation runs separately on `claude-opus-5-5` at `medium` effort.

Each chat keeps one Claude Code process running and reuses it, so replies don't pay a start-up cost, and the conversation lives in a Claude Code session that is resumed after a restart. The CLI runs without your `~/.claude` settings, plugins or MCP servers, and with no built-in tools — only Memclaw's own.

### Cursor

Via [`cursor-sdk`](https://pypi.org/project/cursor-sdk/). Install with `pip install 'memclaw[cursor]'`.

- Generate an API key from the Cursor Dashboard → Integrations (or a team service account key).
- The wizard prompts for `CURSOR_API_KEY` and an optional `CURSOR_MODEL` (defaults to `composer-2.5`).
- Memclaw tools are exposed to the Cursor agent via a long-lived local HTTP MCP server on `127.0.0.1` (default port `17373`, override with `MEMCLAW_MCP_PORT`).
- Requests bill per token against your Cursor plan (or pay-as-you-go overage rates once your plan limit is reached), and the cost is logged with each turn.
- One Cursor SDK bridge process serves all chats, and each chat keeps one local Cursor agent open across turns. The agent stores its conversation under `~/.cursor` and is resumed after a restart.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Embeddings, link summaries and voice transcription |
| `CLAUDE_CODE_OAUTH_TOKEN` | One of these two | Claude subscription token from `claude setup-token` |
| `ANTHROPIC_API_KEY` | One of these two | Anthropic API key (`sk-ant-…`), pay-as-you-go |
| `CLAUDE_MODEL` | Optional | Claude model to run (defaults to `claude-sonnet-5-5`; also set by `/model`) |
| `CLAUDE_EFFORT` | Optional | Effort level: `low`, `medium`, `high`, `xhigh`, or `max` (defaults to `medium` on the default model, otherwise the model decides; also set by `/effort`) |
| `CLAUDE_CONSOLIDATION_MODEL` | Optional | Claude model for memory consolidation (defaults to `claude-opus-5-5`) |
| `CLAUDE_CONSOLIDATION_EFFORT` | Optional | Effort for consolidation (defaults to `medium`) |
| `AGENT_BACKEND` | Optional | Agent SDK to use (defaults to `claude`; set to `cursor` for Cursor SDK) |
| `MEMCLAW_PLATFORM` | Optional | Front-end the bare `memclaw` launches: `telegram`, `whatsapp`, `slack`, or `terminal` (defaults to `terminal`) |
| `CURSOR_API_KEY` | For Cursor backend | Cursor API key from Dashboard → Integrations |
| `CURSOR_MODEL` | For Cursor backend | Cursor model name (defaults to `composer-2.5`; also set by `/model`) |
| `CURSOR_EFFORT` | Optional | Cursor reasoning level, for models that have one (set by `/effort`) |
| `MEMCLAW_MCP_PORT` | For Cursor backend | Local MCP HTTP port (defaults to `17373`) |
| `TELEGRAM_BOT_TOKEN` | For Telegram bot | Your Telegram bot token |
| `ALLOWED_USER_IDS` | For Telegram bot | Comma-separated Telegram user IDs |
| `SLACK_BOT_TOKEN` | For Slack bot | Slack bot token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | For Slack bot | Slack app-level token for Socket Mode (`xapp-...`) |
| `SLACK_ALLOWED_CHANNELS` | For Slack bot | Comma-separated Slack channel IDs (optional) |
| `SLACK_ALLOWED_USERS` | For Slack bot | Comma-separated Slack user IDs (optional — restricts who can DM/mention the bot) |

Search and memory tuning (weights, decay, chunk sizes) is listed in the [Memory Architecture](memory-architecture.md#configuration-reference) doc.

## Directory structure

```
~/.memclaw/
├── .env                   # Saved config and keys
├── MEMORY.md              # Profile + index, written only by consolidation
├── .history/              # MEMORY.md before each consolidation (last 60)
├── AGENTS.md              # Agent instructions + your saved preferences
├── memclaw.db             # SQLite index (embeddings + FTS5, images, reminders)
├── meta.json              # Consolidation progress
├── sessions.json          # Conversation session per chat
├── whatsapp/              # WhatsApp pairing session + received photos (media/)
├── slack/media/           # Photos received via Slack
└── memory/
    ├── 2026-06-15.md      # Daily notes (append-only)
    ├── 2026-06-16.md
    └── ...
```

Memories are plain Markdown; `memclaw index` rebuilds the search index from them. Saved-image records and reminders live only in `memclaw.db`, so back it up along with the rest.
