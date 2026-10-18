# Memclaw Memory Architecture

Memclaw is a personal memory assistant that stores everything you tell it and retrieves it when you need it. It works through Telegram, WhatsApp or Slack bots, or an interactive terminal. Under the hood, it has two distinct layers of memory — short-term and long-term — that work together to give the agent context about who you are and what you've told it.

---

## Short-Term Memory

Short-term memory is the conversation itself: what you and the agent said in the current chat.

### Conversation Sessions

Each chat's conversation lives in the agent backend's own session (for Claude, a Claude Code session; for Cursor, a local Cursor SDK agent), not in a buffer Memclaw re-sends with every message.

**How it works:**

- Every chat gets a session key: `<platform>:<chat id>` for the bots (e.g. `telegram:12345`, `slack:C01ABC`), `cli` for the interactive terminal.
- The Claude backend keeps one connected Claude CLI process per chat and reuses it for every turn, so a message doesn't pay the CLI start-up cost. Turns are serialized, across chats too: the tool executor carries per-turn state (the chat a reminder belongs to, the images to send back), so two chats' turns must not overlap.
- The Cursor backend runs one Cursor SDK bridge process for all chats and keeps one open agent per chat. The agent stores its conversation on disk (under `~/.cursor`, in a store tied to the bridge's working directory, which is pinned to the memory directory). Cursor agents have no system-prompt field, so the system prompt goes at the top of a new agent's first message; later messages carry only the context block and the message.
- The session id is saved in `~/.memclaw/sessions.json` (written atomically). After a restart, the next message in that chat resumes the same session, so the conversation carries on.
- If a session can't be resumed (deleted, created on another machine, ...), a warning is logged and a fresh session starts.
- Each saved session records a **fingerprint** of the code-level instructions (the system-prompt template and the backend). A resumed session keeps its original system prompt, so a session created under different instructions (e.g. after an upgrade that changed the template) is not resumed; the chat starts fresh.
- `/new` (Telegram command; a message of exactly `/new` in Slack, WhatsApp and the terminal) closes that chat's session and forgets its id. Your memories are untouched.
- `/model` and `/effort` (Telegram) switch the model or effort level, and the conversation is kept. Claude sessions reconnect on their next message, resuming the same session with the new settings; Cursor passes the model with every message, so its agents just use it. For Cursor, `/effort` is available only on models that expose a reasoning/effort parameter.

**What this means in practice:**

```
You:       My dog's name is Max.
Assistant: Got it, I'll remember that!
You:       What's my dog called?
Assistant: Your dog's name is Max.
```

The conversation is bounded only by the backend's own context management (e.g. Claude Code compacts long sessions on its own). Use `/new` to start over.

### System Prompt and Per-Message Context

The system prompt is **static for a session**, so the backend can prompt-cache it. It is built when a session is created and contains:

1. **AGENTS.md** — your behavioural instructions (editable via the `update_instructions` tool).
2. Reply-formatting rules and a description of the per-message context block.
3. **MEMORY.md** — your permanent memory (see Permanent Memory in the Prompt below).

Everything that changes per turn goes at the **top of the user message**, in a compact `<context>` block:

- The current local date and time (used for dates and reminders).
- **Relevant memories** — the top 5 search results for the message (hybrid search), each cut to 500 characters. MEMORY.md hits are skipped since MEMORY.md is already in the system prompt. If the search fails (e.g. OpenAI is unreachable), the block is sent without memories rather than failing the message.
- **Delivered reminders** — reminders that fired in this chat since the last message, so the agent has context if you reply to one.
- **Updated AGENTS.md / MEMORY.md** — see below.

The block stays in the session transcript, which is why it is kept small.

### Keeping AGENTS.md and MEMORY.md Current

The session stores a hash of the AGENTS.md and MEMORY.md content it has seen (persisted in `sessions.json`). Before each message the files are hashed again; if either changed (an `update_instructions` call, a consolidation, a manual edit), its new content is included **once** in that message's context block, labelled as superseding the earlier version, and the stored hash is updated. The conversation continues without a reset. AGENTS.md changes the session makes itself during a turn (`update_instructions`) count as seen and aren't sent back. The session never writes MEMORY.md, so any change to it during a turn (a consolidation) is sent with the next message.

Because a resumed session keeps its original system prompt, the stored hashes always describe what the session has actually seen, including across restarts.

## Long-Term Memory

Long-term memory is everything that survives between sessions. It lives on disk as plain Markdown files and in a SQLite database that indexes them for fast retrieval.

### Storage Layout

```
~/.memclaw/
├── MEMORY.md                  # Profile + index, written only by consolidation
├── .history/                  # MEMORY.md before each consolidation (last 60)
├── memclaw.db                 # SQLite: chunks, embeddings, FTS index, cache
├── meta.json                  # Consolidation tracking metadata
├── sessions.json              # Per-chat conversation session ids + hashes
└── memory/
    ├── 2025-03-01.md          # Daily log for March 1
    ├── 2025-03-02.md          # Daily log for March 2
    └── ...
```

**Daily files** (`memory/YYYY-MM-DD.md`) are append-only logs. Every time the agent decides something is worth saving during a conversation, it appends a timestamped entry to today's file. Entries include the content, a type label (note, image, link, voice), and optional tags.

**MEMORY.md** is the overview of you: a holistic **profile** (who you are, the people and pets in your life, work and projects, health, finances in broad strokes, preferences, open threads) plus an **index** that points to the notes holding the details:

```
- Bekele (dog; also Izi, Beka) → memory/2026-04-18.md
## Index
### Credentials
- Lion8 prod login and SSH commands → memory/2026-05-12.md, memory/2026-06-13.md
```

Details — credentials, document numbers, instructions, receipts — stay in the daily files and other notes, which are never deleted. MEMORY.md summarizes and points; it doesn't copy. It has a single writer, consolidation: the agent never writes it directly (`memory_save` has no "permanent" option, and `file_write` refuses MEMORY.md and the daily notes, which it may still read). So the daily files are the complete, append-only record, and whatever a consolidation does to MEMORY.md, it can be rebuilt from them.

### How Memories Get Saved

The agent — not the handler, not a pre-processing step — decides what to save. When you send a text message, voice note, photo, or link, the raw content is passed to the agent with a note saying it hasn't been saved yet. The agent reads it and decides:

- **Save it** if the content is useful (a fact, a note, a decision).
- **Rephrase it** if the verbatim text is messy but the information matters.
- **Skip it** if it's trivial ("hey", "ok", "thanks").

Everything goes to today's daily file — durable facts like your name, preferences or major decisions included. They reach MEMORY.md at the next consolidation, after the day is over; until then they're in today's notes, indexed right away, so search (and the per-message context) finds them.

### Consolidation

Daily files accumulate over time. Consolidation is a periodic process that distills them into MEMORY.md — extracting the important bits and discarding the noise.

**When it runs:**

- Automatically, in the background after a reply, once a past day has notes that aren't consolidated yet — in practice after your first message of the next day you wrote in (days without notes cost nothing; after a few quiet days it catches up in one call). `consolidation_threshold` (default 1) raises the number of pending days to wait for. At most one consolidation runs at a time; a failure is logged and retried after a reply an hour or more later.
- Manually, via the `memclaw consolidate` CLI command (which forces it regardless of count).

Today's file is never consolidated: it is still being written to, and marking it done would skip everything saved later that day. It is picked up once the day is over.

**What it does:**

1. Collects unconsolidated daily files from before today, oldest first, as whole files up to 80,000 characters (~20k tokens). Files that don't fit wait for the next run (a single file larger than the budget is sent truncated, so it can't block consolidation).
2. Reads the current MEMORY.md.
3. Sends them, with today's date and a list of the vault's other notes (name, size, first line), to the agent backend (a short-lived, tool-free call). With Claude it runs on its own model, Opus 5.5 at medium effort by default (`CLAUDE_CONSOLIDATION_MODEL` / `CLAUDE_CONSOLIDATION_EFFORT`; another model gets no effort setting unless one is configured), independent of the chat model. The prompt asks for the profile + index described above, and:
   - details are referenced, never copied — written exactly as `→ memory/YYYY-MM-DD.md` (or `→ todos.md` for another note), pointing to the latest entry when a detail was updated; the one exception is a value that exists only in the current MEMORY.md with no note behind it, which is kept as is (never replaced by a placeholder);
   - only what the sources say; each fact stays with the right person;
   - later entries win, and an explicit correction ("ISPRAVKA", "correction") overrides what it corrects;
   - existing facts and references stay unless updated or expired;
   - left out entirely: expired one-offs (verification codes, past events, finished trips' bookings), links saved to read later (they stay searchable), instructions for the assistant (they belong in AGENTS.md), small talk.

   A wrapping ```` ``` ```` fence in the reply is stripped, and every reference is checked: one to a note that doesn't exist is dropped, one whose note shares no word with the line is logged as unconfirmed.
4. Saves a dated copy of the previous MEMORY.md in `.history/` (the last 60 are kept; the hidden directory isn't indexed or reachable by the file tools), then overwrites MEMORY.md with the result. Lines appended by hand while the backend was working are carried over below the result; if MEMORY.md was otherwise edited meanwhile, nothing is written and consolidation retries later.
5. Records the date of the last file it included in `meta.json` so those files aren't processed again.

The result aims for about 6,000 characters, with absolute dates, in the language MEMORY.md already uses (English when it starts empty).

### Permanent Memory in the Prompt

MEMORY.md goes into the session's system prompt in full (consolidation aims for ~6,000 characters; anything past 20,000 characters is cut with a note pointing the agent to `memory_search`). The system prompt tells the agent that `→ file` lines are references: when you need a detail — an exact value, a number, instructions — it opens the note with `file_read` and answers from it rather than from the summary. Secrets and document numbers therefore aren't sent with every message, only when asked for. Since the system prompt is cached, carrying the whole file costs little. When consolidation rewrites MEMORY.md mid-conversation, the new version reaches the session through the context block as described above.

---

## Search System

Retrieval is how long-term memory becomes useful. When the agent needs to find something — either for context injection or because you explicitly asked — it runs a multi-stage search pipeline.

### Hybrid Search

Every search combines two strategies:

- **Vector search (70% weight):** Each memory chunk has an embedding vector (from OpenAI's `text-embedding-3-small`). The query is embedded too, and cosine similarity finds the semantically closest chunks. This catches meaning even when the words are different — searching "canine" finds "my dog Max".
- **Keyword search (30% weight):** SQLite FTS5 with BM25 scoring. This catches exact matches that vector search might miss — searching "2025-03-01" finds entries with that literal date.

The keyword query is built from the message, not passed to FTS5 as-is: each word is quoted (so punctuation like `What's` or `INV-48213` can't break the query syntax), very common words are dropped, and the rest are OR-ed, so BM25 ranks chunks by how many — and how rare — the matching words are.

Both score lists are normalized to 0–1 (vector scores min-max over the candidates, keyword scores relative to the best BM25 match) and merged with configurable weights.

### Temporal Decay

After merging, scores are adjusted based on age. Recent memories score higher than old ones using exponential decay:

| Age | Score retained |
|-----|---------------|
| Today | 100% |
| 1 week | ~85% |
| 30 days (1 half-life) | ~50% |
| 60 days | ~25% |
| 90 days | ~12.5% |

**Exemptions:** MEMORY.md is evergreen — its content never decays, because it's curated permanent knowledge. Any file that doesn't have a date in its name is also exempt.

Decay can be disabled entirely by setting `decay_half_life_days` to 0. The default half-life is 30 days.

### Deduplication (MMR)

After decay, the results go through Maximal Marginal Relevance filtering. This removes near-duplicate results by penalizing candidates that are too similar to already-selected ones.

It works greedily: the highest-scoring result is always picked first. Then for each remaining candidate, it computes:

```
MMR = 0.7 * relevance - 0.3 * max_similarity_to_selected
```

Similarity is measured with Jaccard distance on word-level tokens. If two chunks share most of their words, the second one gets deprioritized in favor of something different.

This means searching "pizza" returns diverse results (your preference, a restaurant, a recipe) rather than five variations of "I love pizza."

### File Filtering

Search can be restricted to a specific file by passing a `file_filter` string. It filters results by checking if the file path contains the given substring. The filter is applied while merging, before candidates are cut to the oversampling limit, so matches in the filtered file aren't crowded out by other files.

### Full Search Pipeline

For each search call, the pipeline runs in this order:

1. **Embed the query** using OpenAI.
2. **Vector search** — retrieve 3x the requested limit as candidates (all chunks with a file filter).
3. **Keyword search** — retrieve 3x the requested limit as candidates (all chunks with a file filter).
4. **Merge + file filter** — combine and score using weighted blend, keeping only the filtered file if requested.
5. **Temporal decay** — reduce scores for older daily files.
6. **MMR deduplication** — greedily select diverse top-k results.
7. **Return** the final limited result set.

---

## Performance Optimizations

### Embedding Cache

Embeddings are expensive — each one requires an OpenAI API call. To avoid redundant calls, every chunk's text is hashed (SHA-256) before embedding. The hash and resulting embedding are stored in an `embedding_cache` table.

When a file is re-indexed (for example, after appending a new entry), only the chunks whose content hash isn't already cached need to be sent to OpenAI. Unchanged chunks get their embeddings from the local cache instantly.

The cache is also model-aware — if you switch embedding models, cached embeddings from the old model are ignored and new ones are generated.

### Vector Search Matrix Cache

During search, all chunk embeddings need to be loaded from SQLite into a numpy matrix for cosine similarity computation. To avoid doing this on every search, the matrix is cached in memory.

Before each vector search, the system checks the chunk count and the highest chunk id. Re-indexing a file replaces its rows with new ids, so any change — including a rewrite that keeps the same number of chunks, like a consolidation — rebuilds the matrix; otherwise the cached one is reused. This makes repeated searches with no index changes effectively free.

### What Gets Indexed

Every `.md` file under `~/.memclaw/` — MEMORY.md, the daily files, and files the agent creates with `file_write` (e.g. `todos.md`) — except AGENTS.md (instructions, already in the system prompt, not memories) and anything in a hidden directory (e.g. `.cursor/`).

Files are chunked per entry: the `---` line `memory_save` writes after each entry ends a chunk, so a search hit is one entry, without the text or heading of the entry before it. A heading starts a new chunk once the current one is half full. Only an entry longer than ~300 words is split by size, with 60 words of overlap and its heading repeated on each part.

Re-indexing a file fetches its embeddings first and then swaps its chunks in one transaction, so a failed embedding call leaves the previous chunks in place. A file emptied on disk loses its chunks.

### Index Sync Strategy

Rather than scanning the entire filesystem on every search, indexing is event-driven:

- **Write-time indexing:** When the agent saves a memory (or writes a `.md` file with `file_write`), that file is indexed immediately after writing. If indexing fails (e.g. OpenAI is unreachable), the write still counts as saved and the next sync indexes it.
- **Startup sync:** A full filesystem scan runs once when the agent starts, catching any changes made while it was offline (any mtime change, including a file restored with an older mtime). A failed startup sync is logged and doesn't stop the agent.
- **Background sync (bots):** A periodic task runs every 60 seconds to pick up external edits without blocking the search path. Failures are logged and retried on the next tick.
- **Search path:** No sync. Search trusts that the index is up-to-date from the above mechanisms.

---

## Image Memory

Memclaw can also store and retrieve images sent through Telegram, WhatsApp or Slack.

### Saving

When you send a photo, the agent sees the image (via base64), generates a detailed text description of what's in it, and saves (`image_save`):

1. The description as a text entry in the daily file.
2. The platform, its media reference (Telegram `file_id`, WhatsApp media path, Slack file), description, caption, and an embedding vector in the `platform_images` table.

For local images (CLI mode), the file path and any caption are saved the same way, under the platform `local`.

### Retrieval

When you ask "show me that photo of the sunset," the agent runs `image_search`: a vector search over the `platform_images` table using the query's embedding, limited to images saved on the platform you're talking on (a Telegram `file_id` can't be sent through Slack; the terminal searches every platform). It returns the 5 closest images with their ids and descriptions and sends nothing. The agent reads the descriptions and calls `image_send` with the ids of the images that are what you asked for — a similarity score alone can't tell your passport from your ID card when both descriptions mention your name — and each image goes out at most once per reply, even after several searches.

---

## Configuration Reference

All memory-related settings live in `MemclawConfig`:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `memory_dir` | `~/.memclaw` | Root directory for all storage |
| `embedding_model` | `text-embedding-3-small` | OpenAI model for embeddings |
| `embedding_dim` | `1536` | Dimension of embedding vectors |
| `chunk_target_words` | `300` | Target size for text chunks |
| `chunk_overlap_words` | `60` | Overlap between adjacent chunks |
| `vector_weight` | `0.7` | Weight for vector search in hybrid merge |
| `text_weight` | `0.3` | Weight for keyword search in hybrid merge |
| `decay_half_life_days` | `30` | Days until a daily memory's score halves (0 = disabled) |
| `mmr_lambda` | `0.7` | Relevance vs. diversity trade-off in MMR (1.0 = pure relevance) |
| `consolidation_threshold` | `1` | Past days with unconsolidated notes before auto-consolidation (1 = after each day) |
| `claude_consolidation_model` | `claude-opus-5-5` | Claude model for consolidation (`CLAUDE_CONSOLIDATION_MODEL`) |
| `claude_consolidation_effort` | `medium` | Effort for consolidation (`CLAUDE_CONSOLIDATION_EFFORT`) |

---

## Validation

Unit tests (`pytest`) run offline: OpenAI embeddings are replaced by a hashed bag-of-words fake, and the agent runs against a fake backend.

The end-to-end suite in `tests/e2e/` exercises the real stack — OpenAI embeddings, the SQLite index and a real Claude backend — against a throwaway vault in a temp directory (`~/.memclaw` is never touched):

```bash
MEMCLAW_E2E=1 pytest tests/e2e -v
```

It covers retrieval (recall on a seeded vault, keyword search on natural-language questions and identifiers, index freshness after rewrites, what gets indexed, image scoping, file guardrails) and the agent (save then recall after `/new`, a durable fact reaching MEMORY.md through the next day's consolidation, resuming after a restart, saved instructions, an OpenAI outage, consolidation with a concurrent save, the consolidation prompt's promises — details referenced not copied, every reference resolving, a value with no note behind it kept, corrections applied, facts kept with the right person, links, expired one-offs and assistant instructions left out — and the agent following a reference to answer with an exact value when search is down). `MEMCLAW_E2E_MODEL` picks the Claude model and `MEMCLAW_E2E_CONSOLIDATION_MODEL` the consolidation model (both default to `claude-haiku-5-5`; production consolidation uses Opus 5.5).

---

## Data Flow Summary

```
User Message (chat → session key)
│
├─ 1. Look up the chat's saved session (sessions.json; dropped if the fingerprint changed)
├─ 2. Build the <context> block:
│     ├─ Current local time
│     ├─ Reminders delivered since the last message
│     ├─ AGENTS.md / MEMORY.md, if changed since the session last saw them
│     └─ Semantic search for 5 relevant memory chunks
├─ 3. Run the turn in the chat's session (reuse the live client, or resume /
│     start one with the static system prompt: AGENTS.md + rules + MEMORY.md)
│     └─ Agent may call:
│           ├─ memory_save   → append to today's daily file → index
│           ├─ memory_search → run full search pipeline
│           ├─ image_save    → store image metadata
│           ├─ image_search  → vector search over image descriptions (candidates)
│           └─ image_send    → send the chosen images with the reply
├─ 4. Save the session id + AGENTS.md / MEMORY.md hashes
├─ 5. Schedule consolidation in the background (if threshold reached)
│
└─ Return response (+ any found images)
```
