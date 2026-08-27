#!/usr/bin/env bash
# Is the portfolio reporter alive and on schedule?
#
# Run on the server, or remotely:  ssh oci '/home/ubuntu/portfolio/healthcheck.sh'
# Exit code 0 = healthy, 1 = something needs attention.
set -uo pipefail

DIR="${PORTFOLIO_DIR:-/home/ubuntu/portfolio}"
REPORTS="$DIR/reports"
NOW=$(date -u +%s)
PROBLEMS=0

say()  { printf '  %-22s %s\n' "$1" "$2"; }
fail() { printf '  %-22s %s\n' "$1" "$2"; PROBLEMS=$((PROBLEMS+1)); }

echo "Portfolio health  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# --- keepalive ---
if [ -f "$DIR/heartbeat.txt" ]; then
    HB=$(grep -m1 '^last_keepalive=' "$DIR/heartbeat.txt" | cut -d= -f2)
    HB_S=$(date -u -d "$HB" +%s 2>/dev/null || echo 0)
    AGE=$(( (NOW - HB_S) / 60 ))
    if [ "$HB_S" -gt 0 ] && [ "$AGE" -lt 90 ]; then
        say "keepalive"  "ok - ${AGE} min ago"
    else
        fail "keepalive" "STALE - last seen $HB (${AGE} min ago)"
    fi
else
    fail "keepalive" "MISSING - $DIR/heartbeat.txt not found"
fi

# --- latest report ---
if [ -f "$REPORTS/latest.pdf" ]; then
    R_S=$(stat -c %Y "$REPORTS/latest.pdf" 2>/dev/null || echo 0)
    DAYS=$(( (NOW - R_S) / 86400 ))
    if [ "$DAYS" -le 8 ]; then
        say "weekly report" "ok - ${DAYS} day(s) old"
    else
        fail "weekly report" "OVERDUE - ${DAYS} days old"
    fi
    say "report count" "$(ls -1 "$REPORTS"/portfolio-*.pdf 2>/dev/null | wc -l | tr -d ' ') archived"
else
    fail "weekly report" "MISSING - no latest.pdf; has it ever run?"
fi

# --- signal freshness / errors ---
if [ -f "$REPORTS/latest.json" ]; then
    ERRS=$(python3 -c "
import json;d=json.load(open('$REPORTS/latest.json'))
e=d.get('errors') or []
print(('errors: '+','.join(e)) if e else 'no data errors')
print('mode: '+d.get('mode','?'))" 2>/dev/null)
    while IFS= read -r line; do
        case "$line" in
            errors:*) fail "signals" "$line" ;;
            *)        say "signals" "$line" ;;
        esac
    done <<< "$ERRS"
fi

# --- cron ---
if crontab -l 2>/dev/null | grep -q 'run_weekly.sh'; then
    say "cron: report" "installed"
else
    fail "cron: report" "NOT INSTALLED"
fi
if crontab -l 2>/dev/null | grep -q 'keepalive.sh'; then
    say "cron: keepalive" "installed"
else
    fail "cron: keepalive" "NOT INSTALLED"
fi

# --- disk (the only thing that costs money on the free tier) ---
say "disk" "$(df -h / | awk 'NR==2{print $4" free of "$2" ("$5" used)"}')"

echo
if [ "$PROBLEMS" -eq 0 ]; then
    echo "HEALTHY"
else
    echo "$PROBLEMS problem(s) need attention"
fi
exit $(( PROBLEMS > 0 ))
