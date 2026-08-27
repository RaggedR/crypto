#!/usr/bin/env bash
# Generate this week's report. Runs on the OCI instance, under cron.
set -euo pipefail
cd /home/ubuntu/portfolio
mkdir -p reports logs
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting weekly report ==="
./venv/bin/python weekly_report.py \
    --capital "${PORTFOLIO_CAPITAL:-10000}" \
    --outdir /home/ubuntu/portfolio/reports \
    --accounts /home/ubuntu/portfolio/accounts.json
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) done ==="
# Keep the last 52 weekly reports; drop older ones.
ls -1t /home/ubuntu/portfolio/reports/portfolio-*.pdf 2>/dev/null \
    | tail -n +53 | xargs -r rm -f
