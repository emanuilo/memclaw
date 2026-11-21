# Contributing to Memclaw

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/emanuilo/memclaw.git
cd memclaw
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

## Module Structure

CI runs `lint-imports` ([import-linter](https://import-linter.readthedocs.io/)) to keep the modules loosely coupled. Run it locally before opening a PR:

```bash
lint-imports
```

The rules (contracts) live under `[tool.importlinter]` in `pyproject.toml`:

- **Layers**: `cli` → `bot` / `setup` → `agent` → `backends` → `tools` → core modules (`search`, `store`, `reminders`, …) → `index` → `config`. A module may import from layers below it, never above. A new top-level module must be added to a layer.
- **No cycles inside subpackages** such as `memclaw.backends` and `memclaw.bot`.
- **Chat platforms are independent**: the Telegram (`bot/handlers.py`), Slack and WhatsApp handlers never import each other. Put shared code in `memclaw/bot/` or a lower layer.
- **Chat SDKs stay in `memclaw.bot`**: only `memclaw.bot` imports `telegram`, `slack_bolt` or `neonize`.
- **Backends are swappable**: only `memclaw.backends` itself imports the concrete backend modules (`claude.py`, `cursor.py`, …). Everyone else goes through `memclaw.backends` / `memclaw.backends.base`.
- **`memclaw.defaults` stands alone**: the Cursor hook script there is copied out and run on its own, so it can't import memclaw.

Ruff flags imports inside functions (`PLC0415`) so lazy imports stay deliberate and visible. If you need one (e.g. an optional dependency like `cursor-sdk`), add `# noqa: PLC0415 - <reason>`. Hidden cycles are caught by `lint-imports`, which sees function-local imports too.

Existing violations are marked `known violation, to fix:` in `pyproject.toml` and in `# noqa: PLC0415` comments. Don't add new ones; fix the import instead. When you fix one, delete its line or comment (`lint-imports` and ruff both fail on ones that no longer match anything).

If a new dependency between modules is legitimate, move the module to another layer in the same PR; don't add an ignore line.

The main known violation is `backends` importing `setup` (a cycle). It will be fixed by moving `update_env_file` from `setup.py` to `config.py` and `_masked_input` to `prompts.py`.

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add or update tests as needed
5. Run the test suite (`pytest`)
6. Run the linter (`ruff check .`)
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
