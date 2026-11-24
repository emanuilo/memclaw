# Contributing to Memclaw

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/emanuilo/memclaw.git
cd memclaw
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]" --group shape
```

(`--group shape` adds the code-shape checks, see [Code Shape](#code-shape). It needs pip 25.1+ and Python 3.11+. With uv: `uv sync --extra dev --group shape`.)

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

## Running Tests

```bash
pytest
```

All tests should pass before submitting a PR. The test suite mocks external API calls, so no API keys are needed to run tests.

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and small
- Run `ruff check .` before committing

## Code Shape

[code-shape-kit](https://github.com/emanuilo/code-shape-kit) keeps the structure of the code healthy. It checks:

- **Sizes**: files (500 lines), classes (300), folders (12 modules, 4000 lines), nesting of folders.
- **Functions and classes**: complexity, length, number of arguments, variables, attributes and public methods.
- **Coupling**: environment variables or `.env` files read outside `config.py`, the OpenAI SDK or SQLite used outside their modules, private members (`obj._x`) used from outside, copy-pasted code, import cycles, modules that import too many others (more than 8 memclaw modules).
- **Hygiene**: unused code, dependencies in `pyproject.toml` vs. what the code imports, and type errors (mypy).
- **memclaw's layout rules** (see below).

CI runs it on every PR. Locally:

- `code-shape check`: the same check as CI. It fails on a new or worse problem, and also when something got better but the baseline wasn't updated.
- `code-shape report`: writes `code-shape-report.md` with all the numbers, the worst offenders and git-history trends. It never fails.
- `code-shape tighten`: after a fix, records the improvement in the baseline. Commit the baseline together with the fix.

**The baseline.** `.code-shape/baseline.json` lists the problems memclaw already had when the kit was adopted, so only new ones fail. It may only shrink. Never edit it by hand and don't re-run `code-shape baseline` to accept a new problem: fix the code, then run `code-shape tighten`. `# noqa` comments don't silence these checks, and the limits under `[tool.shape]` in `pyproject.toml` aren't raised to make a change pass. If a rule seems wrong for your case, say so in the PR.

**Unused code that isn't.** If the unused-code check (vulture) reports a name that a library uses by reflection (e.g. uvicorn reading `Server.should_exit`), add it to `ignore_names` under `[tool.vulture]` in `pyproject.toml`, with a one-line reason. Make sure nothing could call it first: unused code is usually really unused. Agents ask a human before adding a line.

**Hooks.** Turn on the git hook once per clone, and a commit that fails the check is stopped with the report:

```bash
git config core.hooksPath .githooks
```

Claude Code uses `.claude/settings.json` automatically: when Claude runs `git commit`, the same check runs first and blocks the commit, so Claude sees what to fix. Both check the files in the working tree (stage everything you mean to commit) and take about 15 seconds.

**memclaw's layout rules** are the import-linter contracts under `[tool.importlinter]` in `pyproject.toml`. `code-shape check` runs them (on their own: `lint-imports`):

- **Layers**: `cli` → `setup` → `bot` → `agent` → `backends` → `tools` → `memory` → core modules (`reminders`, `sessions`, `terminal_input`, …) → `openai_client` → `config`. A module may import from layers below it, never above. A new top-level module must be added to a layer. (The terminal prompts, `terminal_input.py`, sit below `backends` rather than inside `setup/` because the backends ask their own setup questions.)
- **No cycles inside subpackages** such as `memclaw.backends` and `memclaw.bot`.
- **Chat platforms are independent**: `bot/telegram.py`, `bot/slack.py`, `bot/whatsapp.py` and `bot/terminal.py` (the terminal chat) never import each other. What the chat apps share (voice to text, link summaries, the prompts, the agent call, reminders) is in `bot/inbound.py`. A new platform is one module with a `run(config)` plus one line in `PLATFORMS` (`bot/__init__.py`); the CLI and the setup wizard start and list platforms only through that registry.
- **Chat SDKs stay in their platform module**: only the three chat-app modules import `telegram`, `slack_bolt` or `neonize`.
- **One OpenAI adapter**: only `openai_client.py` imports `openai`; everyone else calls `OpenAIClient`.
- **Memory through its package**: other code uses what `memclaw.memory` exports (`MemoryStore`, `MemoryIndex`, `HybridSearch`, `Consolidator`, `MemoryVault`, …), not the modules inside (`notes.py`, `index.py`, `search.py`, …). Inside, `vault` (what the one-off `memclaw` commands call) sits on top, `consolidation` and `search` build on `notes` and `index`, never the other way round, and only `index.py` runs SQL on the search index. `memory/consolidation.py` is the only code that writes MEMORY.md; it reaches the LLM through the backend the agent hands it, so it doesn't import `memclaw.backends`.
- **Tools through their package**: other code uses what `memclaw.tools` exports (`TOOL_DEFINITIONS`, `ToolExecutor`, `ToolContext`, `describe_tool_call`). Inside, there is one module per kind of tool (`memory.py`, `images.py`, `files.py`, `reminders.py`), each holding its tools' definitions and code; `catalog.py` lists them in order and `executor.py` runs a call.
- **Setup through its package**: other code uses what `memclaw.setup` exports (`run_setup`, `needs_setup`, `print_logo`, `run_doctor`), not `setup/wizard.py` or `setup/doctor.py`.
- **Backends are swappable**: only `memclaw.backends` itself imports the concrete backends (the `claude/` and `cursor/` packages). Everyone else goes through `memclaw.backends` / `memclaw.backends.base`.
- **`memclaw.defaults` stands alone**: the Cursor hook script there is copied out and run on its own, so it can't import memclaw.

If a new dependency between modules is legitimate, move the module to another layer in the same PR; don't add an ignore line.

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add or update tests as needed
5. Run the test suite (`pytest`)
6. Run the linter (`ruff check .`) and the shape check (`code-shape check`)
7. Commit with a clear message (`git commit -m 'feat: add my feature'`)
8. Push to your fork (`git push origin feature/my-feature`)
9. Open a Pull Request

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) style:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or updating tests
- `chore:` maintenance, dependencies, CI

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if behavior changes
- Link related issues in the PR description

## Reporting Bugs

Open an issue with:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant logs or error messages

## Questions?

Open a [discussion](https://github.com/emanuilo/memclaw/issues) or reach out in an issue.
