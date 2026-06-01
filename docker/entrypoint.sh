#!/usr/bin/env bash
# Memclaw container entrypoint.
#
# Runs as root (set in the Dockerfile) so it can fix ownership of the
# bind-mounted data directory when the host user isn't already uid 1000,
# then drops to the `memclaw` user via gosu before exec'ing the CLI.
#
# Memclaw's own setup wizard fires whenever `$HOME/.memclaw/.env` is
# missing. In a Docker context credentials come from compose env_file
# / `-e` flags, so we create the file (empty is fine, env vars take
# precedence) to short-circuit the wizard.
set -euo pipefail

MEMCLAW_DIR="${MEMCLAW_HOME:-/home/memclaw/.memclaw}"
mkdir -p "$MEMCLAW_DIR"

if [ "$(id -u)" = "0" ]; then
    # Only walk the tree when ownership is actually wrong - avoids a
    # full recursive chown on every restart for steady-state vaults.
    if [ "$(stat -c '%u:%g' "$MEMCLAW_DIR")" != "1000:1000" ]; then
        chown -R memclaw:memclaw "$MEMCLAW_DIR"
    fi
    if [ ! -f "$MEMCLAW_DIR/.env" ]; then
        install -o memclaw -g memclaw -m 600 /dev/null "$MEMCLAW_DIR/.env"
    fi
    exec gosu memclaw "$@"
fi

# Already running as a non-root user (e.g. `docker run -u …`); just
# make sure the .env exists and pass through.
if [ ! -f "$MEMCLAW_DIR/.env" ]; then
    touch "$MEMCLAW_DIR/.env"
fi
exec "$@"
