# Memclaw Memory Architecture

Memclaw is a personal memory assistant that stores everything you tell it and retrieves it when you need it. It works through a Telegram bot or an interactive CLI. Under the hood, it has two distinct layers of memory — short-term and long-term — that work together to give the agent context about who you are and what you've told it.

---

## Short-Term Memory

Short-term memory is the conversation itself: what you and the agent said in the current chat.

### Conversation Sessions

Each chat's conversation lives in the agent backend's own session (for Claude, a Claude Code session), not in a buffer Memclaw re-sends with every message.

**How it works:**

- Every chat gets a session key: `<platform>:<chat id>` for the bots (e.g. `telegram:12345`, `slack:C01ABC`), `cli` for the interactive terminal.
- The Claude backend keeps one connected Claude CLI process per chat and reuses it for every turn, so a message doesn't pay the CLI start-up cost. Turns in the same chat are serialized.
- The session id is saved in `~/.memclaw/sessions.json` (written atomically). After a restart, the next message in that chat resumes the same session, so the conversation carries on.
- If a session can't be resumed (deleted, created on another machine, ...), a warning is logged and a fresh session starts.
- Each saved session records a **fingerprint** of the code-level instructions (the system-prompt template and the backend). Claude Code keeps a session's original system prompt when it resumes, so a session created under different instructions (e.g. after an upgrade that changed the template) is not resumed; the chat starts fresh.
- `/new` (Telegram command; a message of exactly `/new` in Slack, WhatsApp and the terminal) closes that chat's session and forgets its id. Your memories are untouched.
- `/model` and `/effort` (Telegram) switch the model or effort level. Live sessions reconnect on their next message, resuming the same session with the new settings, so the conversation is kept.

**What this means in practice:**

```
You:       My dog's name is Max.
Assistant: Got it, I'll remember that!
You:       What's my dog called?
Assistant: Your dog's name is Max.
```

The conversation is bounded only by the backend's own context management (Claude Code compacts long sessions on its own). Use `/new` to start over.

The Cursor backend has no native sessions yet: as an interim measure it keeps the last 10 messages per chat in memory and folds them into each prompt. That transcript does not survive a restart.

### System Prompt and Per-Message Context

The system prompt is **static for a session**, so the backend can prompt-cache it. It is built when a session is created and contains:

1. **AGENTS.md** — your behavioural instructions (editable via the `update_instructions` tool).
2. Reply-formatting rules and a description of the per-message context block.
3. **MEMORY.md** — your permanent memory (see Permanent Memory in the Prompt below).

Everything that changes per turn goes at the **top of the user message**, in a compact `<context>` block:

- The current local date and time (used for dates and reminders).
- **Relevant memories** — the top 5 search results for the message (hybrid search), each cut to 500 characters. MEMORY.md hits are skipped since MEMORY.md is already in the system prompt.
- **Delivered reminders** — reminders that fired in this chat since the last message, so the agent has context if you reply to one.
- **Updated AGENTS.md / MEMORY.md** — see below.

The block stays in the session transcript, which is why it is kept small.

### Keeping AGENTS.md and MEMORY.md Current

The session stores a hash of the AGENTS.md and MEMORY.md content it has seen (persisted in `sessions.json`). Before each message the files are hashed again; if either changed (an `update_instructions` call, a consolidation, a manual edit), its new content is included **once** in that message's context block, labelled as superseding the earlier version, and the stored hash is updated. The conversation continues without a reset. Changes the session makes itself during a turn (`memory_save` with `permanent=true`, `update_instructions`) count as seen and aren't sent back; a consolidation that rewrites MEMORY.md during a turn is.

Because Claude Code keeps a resumed session's original system prompt, the stored hashes always describe what the session has actually seen, including across restarts.

## Long-Term Memory

Long-term memory is everything that survives between sessions. It lives on disk as plain Markdown files and in a SQLite database that indexes them for fast retrieval.

### Storage Layout

```
~/.memclaw/
├── MEMORY.md                  # Curated permanent knowledge base
├── memclaw.db                 # SQLite: chunks, embeddings, FTS index, cache
├── meta.json                  # Consolidation tracking metadata
├── sessions.json              # Per-chat conversation session ids + hashes
└── memory/
    ├── 2025-03-01.md          # Daily log for March 1
    ├── 2025-03-02.md          # Daily log for March 2
    └── ...
```

**Daily files** (`memory/YYYY-MM-DD.md`) are append-only logs. Every time the agent decides something is worth saving during a conversation, it appends a timestamped entry to today's file. Entries include the content, a type label (note, image, link, voice), and optional tags.

**MEMORY.md** is the permanent knowledge base. It holds curated, structured facts about you — preferences, people, projects, decisions. It is populated either manually (when the agent saves with `permanent=true`) or automatically through consolidation.

### How Memories Get Saved

The agent — not the handler, not a pre-processing step — decides what to save. When you send a text message, voice note, photo, or link, the raw content is passed to the agent with a note saying it hasn't been saved yet. The agent reads it and decides:

- **Save it** if the content is useful (a fact, a note, a decision).
- **Rephrase it** if the verbatim text is messy but the information matters.
- **Skip it** if it's trivial ("hey", "ok", "thanks").

For durable facts like your name, preferences, or major decisions, the agent saves directly to MEMORY.md using `permanent=true`. Everything else goes to the daily file.

### Consolidation

Daily files accumulate over time. Consolidation is a periodic process that distills them into MEMORY.md — extracting the important bits and discarding the noise.

**When it runs:**

- Automatically, in the background after each reply, if 7 or more daily files haven't been consolidated yet. At most one consolidation runs at a time; failures are logged and retried after a later message.
- Manually, via the `memclaw consolidate` CLI command (which forces it regardless of count).

**What it does:**

1. Collects all unconsolidated daily files (up to 30,000 characters).
2. Reads the current MEMORY.md.
3. Sends both to the agent backend (a short-lived, tool-free call) with instructions to extract durable facts, merge with existing content, remove outdated entries, and output a clean structured document.
4. Overwrites MEMORY.md with the result.
5. Records the latest consolidated date in `meta.json` so those files aren't processed again.

The result is a MEMORY.md organized into sections like Preferences, Projects, People, Key Facts, and Decisions — with the most important information at the top.

### Permanent Memory in the Prompt

MEMORY.md goes into the session's system prompt in full (consolidation keeps it under ~5,000 characters; anything past 20,000 characters is cut with a note pointing the agent to `memory_search`). Since the system prompt is cached, carrying the whole file costs little. When consolidation rewrites MEMORY.md mid-conversation, the new version reaches the session through the context block as described above.

---

## Search System

Retrieval is how long-term memory becomes useful. When the agent needs to find something — either for context injection or because you explicitly asked — it runs a multi-stage search pipeline.

### Hybrid Search

Every search combines two strategies:

- **Vector search (70% weight):** Each memory chunk has an embedding vector (from OpenAI's `text-embedding-3-small`). The query is embedded too, and cosine similarity finds the semantically closest chunks. This catches meaning even when the words are different — searching "canine" finds "my dog Max".
- **Keyword search (30% weight):** SQLite FTS5 with BM25 scoring. This catches exact matches that vector search might miss — searching "2025-03-01" finds entries with that literal date.

The two score lists are normalized and merged with configurable weights.

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

Search can be restricted to a specific file by passing a `file_filter` string. It filters results by checking if the file path contains the given substring.

### Full Search Pipeline

For each search call, the pipeline runs in this order:

1. **Embed the query** using OpenAI.
2. **Vector search** — retrieve 3x the requested limit as candidates.
3. **Keyword search** — retrieve 3x the requested limit as candidates.
4. **Merge** — combine and score using weighted blend.
5. **File filter** — restrict to specific file if requested.
6. **Temporal decay** — reduce scores for older daily files.
7. **MMR deduplication** — greedily select diverse top-k results.
8. **Return** the final limited result set.

---

## Performance Optimizations

### Embedding Cache

Embeddings are expensive — each one requires an OpenAI API call. To avoid redundant calls, every chunk's text is hashed (SHA-256) before embedding. The hash and resulting embedding are stored in an `embedding_cache` table.

When a file is re-indexed (for example, after appending a new entry), only the chunks whose content hash isn't already cached need to be sent to OpenAI. Unchanged chunks get their embeddings from the local cache instantly.

The cache is also model-aware — if you switch embedding models, cached embeddings from the old model are ignored and new ones are generated.

### Vector Search Matrix Cache

During search, all chunk embeddings need to be loaded from SQLite into a numpy matrix for cosine similarity computation. To avoid doing this on every search, the matrix is cached in memory.

Before each vector search, the system checks if the chunk count has changed. If it hasn't, the cached matrix is reused. If new chunks were added, the matrix is rebuilt. This makes repeated searches with no index changes effectively free.

### Index Sync Strategy

Rather than scanning the entire filesystem on every search, indexing is event-driven:

- **Write-time indexing:** When the agent saves a memory, that specific file is indexed immediately after writing.
- **Startup sync:** A full filesystem scan runs once when the agent starts, catching any changes made while it was offline.
- **Background sync (Telegram bot):** A periodic task runs every 60 seconds to pick up external edits without blocking the search path.
- **Search path:** No sync. Search trusts that the index is up-to-date from the above mechanisms.

---

## Image Memory

Memclaw can also store and retrieve images, primarily through the Telegram bot.

### Saving

When you send a photo on Telegram, the agent sees the image (via base64), generates a detailed text description of what's in it, and saves:

1. The description as a text entry in the daily file.
2. The Telegram `file_id`, description, caption, and an embedding vector in the `telegram_images` table.

For local images (CLI mode), the file path and any caption are saved as a text memory entry.

### Retrieval

When you ask "show me that photo of the sunset," the agent runs a vector search over the `telegram_images` table using the query's embedding. Matching images are returned by `file_id` and sent back through Telegram automatically.

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
| `consolidation_threshold` | `7` | Number of unconsolidated daily files before auto-consolidation |

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
│           ├─ memory_save   → write to daily file or MEMORY.md → index
│           ├─ memory_search → run full search pipeline
│           ├─ image_save    → store image metadata
│           └─ image_search  → vector search over image descriptions
├─ 4. Save the session id + AGENTS.md / MEMORY.md hashes
├─ 5. Schedule consolidation in the background (if threshold reached)
│
└─ Return response (+ any found images)
```
