# syntax=docker/dockerfile:1.7
#
# Memclaw - production image for VPS deployment.
#
# Two-stage build: stage 1 resolves Python deps with uv into a venv,
# stage 2 ships only the venv + a slim runtime (Python + Node for the
# Claude CLI + libmagic for WhatsApp/neonize).

# ---------- 1. dependency build ----------
FROM python:3.12-slim-trixie AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Static-linked uv binary, copied from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /usr/local/bin/

WORKDIR /app

# Resolve dependencies first - this layer caches as long as the lockfile
# doesn't change. README.md is required because pyproject.toml references it.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra cursor

# Now copy the source and install the project itself into the venv.
COPY memclaw ./memclaw
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cursor


# ---------- 2. runtime ----------
FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/memclaw \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH" \
    # Memclaw uses HOME/.memclaw; making this explicit documents the contract
    # for anyone overriding the volume mount.
    MEMCLAW_HOME=/home/memclaw/.memclaw

# System deps:
#   libmagic1 - python-magic / neonize (WhatsApp backend)
#   tini      - PID 1 signal handling, clean Ctrl-C and SIGTERM
#   gosu      - entrypoint chowns the bind mount as root then drops to
#               the memclaw user (standard Postgres-image pattern)
#   nodejs    - the claude-agent-sdk shells out to the Claude Code CLI
#   curl/ca-certificates - needed temporarily for the NodeSource setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gosu \
        libmagic1 \
        tini \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    # gnupg/dirmngr only existed to verify the NodeSource apt signature;
    # keeping them in the final image would carry ~12 unfixed GnuPG CVEs
    # for no runtime benefit.
    && apt-get purge -y curl gnupg gnupg-l10n dirmngr \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/* /root/.npm

# Non-root runtime user. Fixed UID/GID so bind-mounted host volumes can be
# chowned predictably (`chown -R 1000:1000 ./data`).
RUN groupadd --system --gid 1000 memclaw \
    && useradd  --system --uid 1000 --gid memclaw \
                --create-home --home-dir /home/memclaw \
                --shell /bin/bash memclaw

COPY --from=builder --chown=memclaw:memclaw /app /app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /home/memclaw/.memclaw \
    && chown -R memclaw:memclaw /home/memclaw

# Entrypoint starts as root so it can chown the bind-mounted data dir
# when the host user isn't already uid 1000, then drops to `memclaw`
# via gosu. See docker/entrypoint.sh.
WORKDIR /home/memclaw

# Persistent state: MEMORY.md, memclaw.db, daily notes, images, bot sessions.
VOLUME ["/home/memclaw/.memclaw"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["memclaw"]
