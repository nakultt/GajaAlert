#!/usr/bin/env bash
# start-uno-q.sh — run this ON the UNO Q device.
#
# Thin wrapper around live_elephant_detect.py so you don't have to remember
# the flags/env vars each time. Reports detections to the Gaja-alert pipeline
# running on the Snapdragon laptop (see Gaja-alert/scripts/start-gaja.ps1).
#
# Usage:
#   ./start-uno-q.sh <laptop-ip>
#   LAPTOP_HOST=<laptop-ip> ./start-uno-q.sh
#
# Env vars (override the positional arg above):
#   LAPTOP_HOST         laptop's IP/hostname (required)
#   LAPTOP_SENSOR_PORT  laptop's sensor WS port (default 9000, matches Gaja-alert's GAJA_SENSOR_PORT)

set -euo pipefail

LAPTOP_HOST="${1:-${LAPTOP_HOST:-}}"
export LAPTOP_SENSOR_PORT="${LAPTOP_SENSOR_PORT:-9000}"

if [ -z "$LAPTOP_HOST" ]; then
    echo "usage: $0 <laptop-ip>" >&2
    echo "   or: LAPTOP_HOST=<laptop-ip> $0" >&2
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[start-uno-q] laptop: $LAPTOP_HOST:$LAPTOP_SENSOR_PORT"
export LAPTOP_HOST
exec python3 live_elephant_detect.py --laptop-host "$LAPTOP_HOST"
