#!/usr/bin/env bash
# Memclaw container entrypoint.
#
# Memclaw's CLI runs an interactive setup wizard whenever
# `$HOME/.memclaw/.env` is missing. In a Docker context the credentials
# come from the compose env_file / `-e` flags, so we create the file
# (empty is fine, env vars take precedence) to short-circuit the wizard.
set -euo pipefail

MEMCLAW_DIR="${MEMCLAW_HOME:-$HOME/.memclaw}"
mkdir -p "$MEMCLAW_DIR"

if [ ! -f "$MEMCLAW_DIR/.env" ]; then
    touch "$MEMCLAW_DIR/.env"
fi

exec "$@"
