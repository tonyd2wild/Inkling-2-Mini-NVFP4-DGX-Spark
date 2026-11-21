#!/usr/bin/env bash
# Stop and remove the Inkling 2 Mini container (any profile). Weights and image stay on disk.
NAME="${NAME:-inkling2mini}"
docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME"
free -g
