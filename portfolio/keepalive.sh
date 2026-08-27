#!/usr/bin/env bash
# Keep an Oracle Always Free instance from being reclaimed as "idle".
#
# Oracle may reclaim an Always Free compute instance when, across a
# 7-day window, ALL of the following are true:
#     - CPU    95th percentile < 20%
#     - Network 95th percentile < 20%
#     - Memory 95th percentile < 20%   (Ampere A1 shapes only)
#
# Because the test is a conjunction, defeating ONE criterion is enough.
# CPU is the cheapest to move, so that is what we do: burn one core at
# the lowest possible priority for a few minutes each half hour. That
# puts well over 5% of samples above the threshold, which is what the
# 95th percentile needs.
#
# 'nice -n 19' plus 'ionice -c3' means this yields to absolutely
# everything else on the box -- the AI agents and the Gateway app never
# wait on it.
#
# Cron: */30 * * * * /home/ubuntu/portfolio/keepalive.sh
set -euo pipefail

BURN_SECONDS="${KEEPALIVE_BURN_SECONDS:-300}"
STATE_DIR="${KEEPALIVE_STATE_DIR:-/home/ubuntu/portfolio}"
HEARTBEAT="$STATE_DIR/heartbeat.txt"

mkdir -p "$STATE_DIR"

# Heartbeat first, so the timestamp is written even if the burn is killed.
{
    echo "last_keepalive=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uptime=$(uptime -p 2>/dev/null || true)"
    echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "burn_seconds=$BURN_SECONDS"
} > "$HEARTBEAT"

# One core, lowest priority, hard timeout. Prefer stress-ng if present.
if command -v stress-ng >/dev/null 2>&1; then
    nice -n 19 ionice -c3 stress-ng --cpu 1 --timeout "${BURN_SECONDS}s" \
        --quiet >/dev/null 2>&1 || true
else
    # Portable fallback: a busy loop that cannot outlive the timeout.
    nice -n 19 ionice -c3 timeout "${BURN_SECONDS}" \
        bash -c 'while :; do :; done' >/dev/null 2>&1 || true
fi

echo "burn_completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$HEARTBEAT"
