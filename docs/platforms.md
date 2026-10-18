# Messaging Platforms

Memclaw runs as a Telegram, WhatsApp or Slack bot, or in your terminal. The setup wizard (`memclaw`, or `memclaw configure` later) asks which one and prompts only for that platform's tokens. All platforms share the same agent, memories and search index.

Just talk to it — no commands needed. Send text, photos, voice messages or links, or ask to be reminded later; the agent decides whether to store, search, send back images or schedule a reminder.

Each chat is one ongoing conversation that survives restarts. Send `/new` to start a fresh one (your memories are kept). On Telegram it's a menu command; in Slack, WhatsApp and the terminal send a message that is exactly `/new`.

## What it handles

| Message type | What happens |
|-------------|-------------|
| **Text** | The agent decides: store as a memory, search existing memories, or both. |
| **Link** | URLs in any message are fetched, summarized and indexed alongside the surrounding text. |
| **Photo** | The agent describes it, stores the description and indexes it. Ask for it later and the photo is sent back. |
| **Voice / Audio** | Transcribed with Whisper and handled like text. |
| **Reminders** | One-shot (`remind me tomorrow at 9am to call Alex`) or recurring (`remind me every 5 hours to drink water`), delivered to the same chat. List with *show my reminders*, cancel by id. |

## Telegram

Shows a **typing indicator** while processing, plus a silent "Working… step N" message that follows the agent's tool calls (searching, saving, reminders) and disappears once the reply arrives.

Commands: `/new` starts a fresh conversation, `/model` shows the available models as buttons (tap one to switch, or send `/model <id>`), and `/effort` does the same for the reasoning effort. Model and effort changes are saved to `~/.memclaw/.env` and keep the current conversation. With Cursor, `/effort` only applies to models that expose a reasoning setting.

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Get your Telegram user ID (e.g. via [@userinfobot](https://t.me/userinfobot)).
3. Run `memclaw`, pick **Telegram**, and enter the token and your user ID when asked.

## WhatsApp

Uses your **personal WhatsApp account** via WhatsApp Web pairing — no Meta Business account, no webhooks, no public server. Powered by [`neonize`](https://github.com/krypton-byte/neonize) (whatsmeow under the hood).

### Prerequisite: libmagic

neonize depends on `python-magic`, which needs the `libmagic` system library:

- macOS: `brew install libmagic`
- Debian/Ubuntu: `sudo apt install libmagic1`
- Fedora/RHEL: `sudo dnf install file-libs`

### Setup

Run `memclaw` and pick **WhatsApp**. A QR code is printed to your terminal. On your phone: **Settings → Linked Devices → Link a Device**, and scan it. The session persists under `~/.memclaw/whatsapp/` so you only pair once.

Only messages you send to yourself (WhatsApp's "Message Yourself" chat) are processed. DMs from other people and group messages are ignored.

> **Reminders caveat:** because the bot runs as you and replies in your own "Message Yourself" chat, WhatsApp doesn't push a notification when a reminder fires — the message just appears silently in the chat. Delivering reminders from a dedicated Memclaw number (so you do get a notification) is coming soon.

## Slack

Connects via **Socket Mode** (WebSocket) — no public URL or webhook server needed.

### Setup (from manifest — recommended)

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From a manifest**, pick your workspace, and paste:

    ```yaml
    display_information:
      name: Memclaw
      description: Your personal memory vault, powered by AI.
      background_color: "#1a1a1a"
    features:
      bot_user:
        display_name: Memclaw
        always_online: true
      app_home:
        home_tab_enabled: false
        messages_tab_enabled: true
        messages_tab_read_only_enabled: false
    oauth_config:
      scopes:
        bot:
          - app_mentions:read
          - channels:history
          - chat:write
          - files:read
          - files:write
          - im:history
          - im:read
          - im:write
    settings:
      event_subscriptions:
        bot_events:
          - app_mention
          - message.im
      interactivity:
        is_enabled: false
      socket_mode_enabled: true
    ```

2. **Install to Workspace** and copy the **Bot Token** (`xoxb-...`).
3. Under **Basic Information → App-Level Tokens**, generate a token with `connections:write` scope and copy it (`xapp-...`).
4. Run `memclaw`, pick **Slack**, and enter both tokens.

DM the bot directly or mention it in channels (`@Memclaw save this...`). Optionally restrict it to specific channels with `SLACK_ALLOWED_CHANNELS`, or to specific users with `SLACK_ALLOWED_USERS` (find your Slack ID via **Profile → ••• → Copy member ID**).

## Terminal

Pick **Terminal** in the wizard (the default), then:

```
> Just had coffee with Alex. She's moving to Berlin next month for a new role at Stripe.
✓ Memory saved

> Save the image at ~/photos/whiteboard.jpg — it's our sprint planning board
✓ Image saved: A whiteboard showing a sprint planning layout with colorful sticky notes...

> /new
Started a new conversation.

> /quit
```

### Direct commands

These work without the agent — only the OpenAI key is needed.

```bash
memclaw save "Meeting at 3pm with the design team about the rebrand"
memclaw search "design team meetings"
memclaw consolidate   # distill finished days' notes into MEMORY.md now
memclaw index         # rebuild the search index
memclaw status        # vault overview
memclaw doctor        # check your OpenAI key can reach every model Memclaw uses
```

If voice messages aren't transcribed or link summaries come back empty, run `memclaw doctor`. It probes `text-embedding-3-small` (search), `gpt-5-mini` (link summaries) and `whisper-1` (voice) and tells you which ones your OpenAI project can use. The same check runs at the end of the setup wizard.
