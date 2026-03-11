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
