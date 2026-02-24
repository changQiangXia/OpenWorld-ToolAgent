#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command...>"
  exit 2
fi

MAX_RETRIES="${MAX_RETRIES:-2}"
ATTEMPT=0

while true; do
  if "$@"; then
    exit 0
  fi

  ATTEMPT=$((ATTEMPT + 1))
  if [ "$ATTEMPT" -gt "$MAX_RETRIES" ]; then
    echo "Command failed after $((MAX_RETRIES + 1)) attempts."
    exit 1
  fi

  echo "Attempt ${ATTEMPT} failed. Retrying in 2 seconds..."
  sleep 2
done
