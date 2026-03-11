# Memclaw Memory Architecture

Memclaw is a personal memory assistant that stores everything you tell it and retrieves it when you need it. It works through a Telegram bot or an interactive CLI. Under the hood, it has two distinct layers of memory — short-term and long-term — that work together to give the agent context about who you are and what you've told it.

---

## Short-Term Memory

Short-term memory covers anything the agent needs to remember within a single session but that doesn't persist across restarts.

### Conversation History

The agent keeps a rolling buffer of recent messages — both yours and its own responses. This lets it follow multi-turn conversations without losing the thread.

**How it works:**

- Each time you send a message, it's appended to an in-memory list along with a timestamp.
- When the agent responds, that response is appended too.
- The buffer is capped at the last 10 exchanges (20 entries: 10 from you, 10 from the agent). Older entries are dropped.
- The full history is formatted and included in the system prompt so the agent can see what was said earlier in the session.

**What this means in practice:**

```
You:       My dog's name is Max.
Assistant: Got it, I'll remember that!
You:       What's my dog called?
Assistant: Your dog's name is Max.
```

Without this buffer, the second question would get "I don't know" because each message was previously independent.

**Limitations:**

- The buffer resets when the process restarts. It is not written to disk.
- For images, the history stores `[User sent a photo]` as a placeholder — not the actual image data.
- The default limit of 10 pairs is configurable via `conversation_history_limit`.

### Memory Context Injection

Before every response, the agent builds a context block from long-term storage and injects it into the system prompt. This is ephemeral — it's assembled fresh for each message based on what seems relevant.

The context has two parts:

1. **Permanent memory (MEMORY.md)** — always included (see Smart Context Strategy below).
2. **Relevant memories** — the top 10 search results from the full memory index, matched against the current message using hybrid search.

This means the agent always has access to your key facts and any memories related to what you're currently talking about, even if you didn't explicitly ask to search.

---

## Long-Term Memory

Long-term memory is everything that survives between sessions. It lives on disk as plain Markdown files and in a SQLite database that indexes them for fast retrieval.

### Storage Layout

```
~/.memclaw/
├── MEMORY.md                  # Curated permanent knowledge base
├── memclaw.db                 # SQLite: chunks, embeddings, FTS index, cache
├── meta.json                  # Consolidation tracking metadata
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

- Automatically, before each message, if 7 or more daily files haven't been consolidated yet.
- Manually, via the `memclaw consolidate` CLI command (which forces it regardless of count).

**What it does:**

1. Collects all unconsolidated daily files (up to 30,000 characters).
2. Reads the current MEMORY.md.
3. Sends both to Claude with instructions to extract durable facts, merge with existing content, remove outdated entries, and output a clean structured document.
4. Overwrites MEMORY.md with the result.
5. Records the latest consolidated date in `meta.json` so those files aren't processed again.

The result is a MEMORY.md organized into sections like Preferences, Projects, People, Key Facts, and Decisions — with the most important information at the top.

### Smart Context Strategy

When MEMORY.md is included in the system prompt, it uses an adaptive strategy rather than a hard character cutoff:

- **Small file (under 4,000 characters):** Included in full. Nothing is lost.
- **Large file (over 4,000 characters):** The first 2,000 characters are always included (this is where consolidation places the most important content). Then a semantic search runs against just the MEMORY.md chunks using the current message as query, and the top 3 matching sections are appended.

This means a 10,000-character MEMORY.md doesn't lose relevant information — it just includes the top section plus the parts most related to what you're currently asking about.

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

Search can be restricted to a specific file by passing a `file_filter` string. Internally this is used to search only within MEMORY.md when building the smart context strategy. It filters results by checking if the file path contains the given substring.

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
| `conversation_history_limit` | `10` | Number of message pairs to keep in session buffer |
| `consolidation_threshold` | `7` | Number of unconsolidated daily files before auto-consolidation |

---

## Data Flow Summary

```
User Message
│
├─ 1. Consolidation check (merge daily files → MEMORY.md if threshold reached)
├─ 2. Append to conversation history buffer
├─ 3. Build context:
│     ├─ Include MEMORY.md (full or smart-truncated)
│     └─ Semantic search for 10 relevant memory chunks
├─ 4. Format history for system prompt
├─ 5. Send to Claude with context + history + tools
│     └─ Agent may call:
│           ├─ memory_save   → write to daily file or MEMORY.md → index
│           ├─ memory_search → run full search pipeline
│           ├─ image_save / telegram_image_save → store image metadata
│           └─ image_search  → vector search over image descriptions
├─ 6. Append assistant response to history buffer
├─ 7. Trim history to last N pairs
│
└─ Return response (+ any found images for Telegram)
```
