# Running Memclaw on a VPS with Docker

Memclaw runs as a single long-lived bot process. Every supported front-end
(Telegram, Slack, WhatsApp) is outbound - the container needs no published
ports, just OpenAI + agent-backend credentials and bot tokens.

## What's in the image

- `python:3.12-slim-bookworm` base
- Memclaw and its Python deps, installed from `uv.lock` into `/app/.venv`
- Node.js 24 (active LTS) + `@anthropic-ai/claude-code` (the Claude backend shells out to it)
- `libmagic1` (required by `neonize` / `python-magic` for the WhatsApp backend)
- A non-root user `memclaw` (uid 1000) whose `~/.memclaw` is the persistent
  data volume

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env`. The minimum for a Telegram deployment looks like this:

```dotenv
MEMCLAW_PLATFORM=telegram

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

TELEGRAM_BOT_TOKEN=123456:ABC...
ALLOWED_USER_IDS=123456789

# Local timezone for reminders and daily-note dates (defaults to UTC)
TZ=Europe/Belgrade
```

Slack and WhatsApp work the same way - set `MEMCLAW_PLATFORM` and fill in
that platform's tokens. See the project README for the full list of env vars.

> The container's entrypoint creates an empty `~/.memclaw/.env` on first
> boot to skip Memclaw's interactive setup wizard. Credentials are taken
> from the env vars supplied by Docker, not from that file.

> **Set `TZ` to your local timezone** (an IANA name such as
> `Europe/Belgrade` or `America/New_York`). Memclaw reads the system clock,
> so if `TZ` is unset the container runs on UTC: absolute reminders ("remind
> me at 9am") fire on UTC wall-clock time and daily notes roll over at UTC
> midnight. The image bundles `tzdata`, so any IANA zone works and the
> CET/CEST daylight-saving switch is handled automatically.

## 2. Build and start

```bash
docker compose up -d --build
docker compose logs -f memclaw
```

The image takes a few minutes the first time (Node + Claude CLI install).
Subsequent rebuilds reuse the dependency layer.

## 3. WhatsApp first-run pairing

WhatsApp needs you to scan a QR code once. Run the container in the
foreground for the initial pairing, then bring it up detached:

```bash
docker compose run --rm memclaw     # scan the QR, wait for "linked"
# Ctrl-C once paired
docker compose up -d
```

The session lives at `/home/memclaw/.memclaw/whatsapp/session.db` inside
the named volume, so you only pair once.

## 4. Data persistence

The compose file bind-mounts `./data` on the host to
`/home/memclaw/.memclaw` in the container so files survive image
rebuilds and can be backed up / inspected directly. It holds:

- `MEMORY.md`, `AGENTS.md`, and `memory/YYYY-MM-DD.md` daily notes
- `memclaw.db` - the SQLite index (embeddings + FTS5)
- `images/` - saved photos
- `whatsapp/session.db` - the paired WhatsApp session
- `slack/`, `bot.log`, `whatsapp.log`, `slack.log`

The container runs as uid:gid `1000:1000`. The entrypoint starts as
root, chowns `./data` to that uid on first boot (only when ownership
is actually wrong, so steady-state restarts skip the walk), and drops
to the `memclaw` user via `gosu` before exec'ing the CLI - so you do
*not* need to `chown` the directory yourself, even on a host where
your user isn't 1000.

`data/` is gitignored. Prefer a named volume instead? Swap the bind
mount for `memclaw-data:/home/memclaw/.memclaw` and add a top-level
`volumes: { memclaw-data: {} }` block.

## 5. Common operations

```bash
# View status from inside the container
docker compose exec memclaw memclaw status

# Search memories without touching the agent
docker compose exec memclaw memclaw search "berlin"

# Check OpenAI model access
docker compose exec memclaw memclaw doctor

# Rebuild the search index
docker compose exec memclaw memclaw index

# Update to a new image
git pull
docker compose up -d --build
```

## 6. Resource notes

- The container has no exposed ports - run it behind your VPS firewall
  as-is. No reverse proxy needed.
- Memory footprint is modest (a few hundred MB), but the SQLite index
  grows with your memory vault, so back the volume up periodically.
- The Claude backend (`AGENT_BACKEND=claude`, the default) spawns the
  Claude Code CLI per turn. The Cursor backend (`AGENT_BACKEND=cursor`)
  is pure-Python and slightly leaner if you don't need Claude.
