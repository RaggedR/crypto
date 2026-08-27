#!/usr/bin/env bash
# Install the recycling-portfolio reporter on the OCI instance.
#
# Run this FROM YOUR MAC. It copies the scripts to the server, builds a
# virtualenv, runs one report to prove the install works, and installs a
# weekly cron job. Idempotent: safe to run repeatedly.
#
#   bash install_oci.sh
#
# Assumes an ssh alias 'oci' exists (see oci.pdf section on SSH).

set -euo pipefail

HOST="${OCI_HOST:-oci}"
REMOTE_DIR="/home/ubuntu/portfolio"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Checking SSH access to '$HOST'"
ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" 'echo "    connected as $(whoami)@$(hostname)"'

echo "==> Creating $REMOTE_DIR"
ssh "$HOST" "mkdir -p $REMOTE_DIR/reports $REMOTE_DIR/logs"

echo "==> Copying scripts"
scp -q "$LOCAL_DIR/signals.py" "$LOCAL_DIR/weekly_report.py" \
       "$LOCAL_DIR/cap_sweep.py" "$LOCAL_DIR/run_weekly.sh" \
       "$LOCAL_DIR/keepalive.sh" "$LOCAL_DIR/healthcheck.sh" \
       "$HOST:$REMOTE_DIR/"
ssh "$HOST" "chmod +x $REMOTE_DIR/run_weekly.sh $REMOTE_DIR/keepalive.sh $REMOTE_DIR/healthcheck.sh"

echo "==> Building virtualenv (aarch64 wheels; this takes a few minutes)"
ssh "$HOST" "bash -s" <<'REMOTE'
set -euo pipefail
cd /home/ubuntu/portfolio
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev >/dev/null
if [ ! -d venv ]; then python3 -m venv venv; fi
./venv/bin/pip install --quiet --upgrade pip wheel
./venv/bin/pip install --quiet numpy pandas matplotlib yfinance
echo "    installed: $(./venv/bin/python -c 'import numpy,pandas,matplotlib,yfinance as y; print("numpy",numpy.__version__,"pandas",pandas.__version__,"yfinance",y.__version__)')"
REMOTE

echo "==> Running one report now (proves the install works)"
ssh "$HOST" "cd $REMOTE_DIR && ./run_weekly.sh"

echo "==> Starting keepalive once, to prove it runs"
ssh "$HOST" "KEEPALIVE_BURN_SECONDS=5 $REMOTE_DIR/keepalive.sh && cat $REMOTE_DIR/heartbeat.txt"

echo "==> Installing cron jobs (weekly report + Oracle idle keepalive)"
ssh "$HOST" "bash -s" <<'REMOTE'
set -euo pipefail
REPORT='15 21 * * 0 /home/ubuntu/portfolio/run_weekly.sh >> /home/ubuntu/portfolio/logs/cron.log 2>&1'
ALIVE='*/30 * * * * /home/ubuntu/portfolio/keepalive.sh >> /home/ubuntu/portfolio/logs/keepalive.log 2>&1'
TMP=$(mktemp)
crontab -l 2>/dev/null \
  | grep -v 'portfolio/run_weekly.sh' \
  | grep -v 'portfolio/keepalive.sh' > "$TMP" || true
echo '# Recycling portfolio: weekly report + keep instance from being reclaimed' >> "$TMP"
echo "$REPORT" >> "$TMP"
echo "$ALIVE" >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "    cron installed:"
crontab -l | grep -A2 'Recycling portfolio'
REMOTE

echo "==> Health check"
ssh "$HOST" "$REMOTE_DIR/healthcheck.sh" || true

echo
echo "==> Done. Fetch the latest report with:"
echo "      bash $LOCAL_DIR/fetch_report.sh"
