"""What a chat session is started with: the system prompt, and the
fingerprint that tells whether a saved session may be resumed.

The system prompt is AGENTS.md, Memclaw's fixed rules (reply formatting, the
per-message context block, how to read MEMORY.md's references) and
MEMORY.md. It is static for a session, so the backend can prompt-cache it;
when AGENTS.md or MEMORY.md change later, the agent sends the new content in
the next message's context block instead (see `memclaw.agent`).
"""

from __future__ import annotations

import hashlib

from .config import MemclawConfig

# Static for the whole session. Changing this template changes the session
# fingerprint, so existing sessions start fresh instead of being resumed.
_SYSTEM_PROMPT_TEMPLATE = """\
{agent_instructions}

=== REPLY FORMATTING ===
Replies are delivered to messaging apps with limited markdown support. Use \
ONLY this minimal syntax — anything else leaks as literal characters:
- Bold: `*bold*` (single asterisk). NEVER use `**double asterisks**`.
- Italic: `_italic_`.
- Bullet lists: plain `- item` on its own line.
- Paragraphs: separate with a blank line.
- Headings (`#`, `##`, ...) are NOT supported — use a bold line on its own \
(e.g. `*Section name*`) followed by a blank line instead.
- No backticks or fenced code blocks.
- No `[label](url)` links — write the bare URL.

=== PER-MESSAGE CONTEXT ===
Each user message starts with a <context> block added by Memclaw, not typed \
by the user. It holds the current local time (use it for dates and \
reminders), memories that may be relevant to that message, reminders that \
were delivered to the user since their last message, and the new content of \
AGENTS.md or MEMORY.md when either file changes — such an update supersedes \
the version in this system prompt and any earlier update. Don't mention the \
block itself to the user.

=== PERMANENT MEMORY (MEMORY.md) ===
An overview of the user. Lines ending in "→ memory/2026-04-18.md" (or \
another file) point to the notes holding the details: when the user needs \
a detail — an exact value, a number, instructions — open the note with \
file_read and answer from it, rather than from the summary. memory_search \
finds anything else.

{permanent_memory}

IMPORTANT: When the user gives you a behavioural instruction (e.g. "always respond \
in Spanish", "be more formal", "never use emojis"), you MUST call the \
update_instructions tool to save it. These are rules you should follow in every \
future conversation.
"""

# Consolidation aims for ~6,000 characters of MEMORY.md;
# this cap only guards against a runaway file.
_MEMORY_PROMPT_CHARS = 20000


def load_agent_instructions(config: MemclawConfig) -> str:
    """AGENTS.md, the user's behavioural instructions (or a default)."""
    agent_file = config.agent_file
    if agent_file.exists():
        return agent_file.read_text().strip()
    return "You are Memclaw, a personal memory assistant."


def load_permanent_memory(config: MemclawConfig) -> str:
    """MEMORY.md as it goes into the prompt: "(empty)" when there is none,
    cut at `_MEMORY_PROMPT_CHARS`."""
    memory = config.memory_file.read_text().strip() if config.memory_file.exists() else ""
    if not memory:
        return "(empty)"
    if len(memory) > _MEMORY_PROMPT_CHARS:
        return memory[:_MEMORY_PROMPT_CHARS] + "\n\n[... truncated; use memory_search for more]"
    return memory


def build_system_prompt(agent_instructions: str, permanent_memory: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_instructions=agent_instructions,
        permanent_memory=permanent_memory,
    )


def content_hash(text: str) -> str:
    """Short hash of a file's content, to tell whether a session has seen it."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def instructions_fingerprint(backend_name: str) -> str:
    """Hash of the code-level instructions a session is started with.

    Claude Code keeps a session's original system prompt when it is resumed,
    so a session started under a different template (or backend) must not be
    resumed. AGENTS.md / MEMORY.md are tracked separately per session and
    sent as updates instead, so editing them doesn't reset conversations.
    """
    return content_hash(f"{backend_name}\0{_SYSTEM_PROMPT_TEMPLATE}")
