#!/usr/bin/env bash
# Pull the latest report off the OCI instance and open it.
#   bash fetch_report.sh            # latest
#   bash fetch_report.sh 2026-W35   # a specific week
set -euo pipefail
HOST="${OCI_HOST:-oci}"
DEST="${REPORT_DEST:-$HOME/git/andy/crypto/portfolio/reports}"
mkdir -p "$DEST"

if [ $# -ge 1 ]; then REMOTE="portfolio-$1.pdf"; else REMOTE="latest.pdf"; fi

scp -q "$HOST:/home/ubuntu/portfolio/reports/$REMOTE" "$DEST/$REMOTE"
scp -q "$HOST:/home/ubuntu/portfolio/reports/latest.json" "$DEST/latest.json" 2>/dev/null || true

echo "Fetched $DEST/$REMOTE"
python3 -c "
import json,sys
try: d=json.load(open('$DEST/latest.json'))
except Exception: sys.exit()
s=d.get('stats') or {}
print(f\"  week {d['week']}  mode={d['mode']}\")
if s: print(f\"  NAV \${s['nav']:,.2f}  week {(s.get('week_return') or 0)*100:+.2f}%  total {s['total_return']*100:+.2f}%  maxDD {s['max_dd']*100:.1f}%\")
print('  weights: ' + ', '.join(f\"{k} {v*100:.0f}%\" for k,v in d['weights'].items() if v>0) + f\", cash {d['cash']*100:.0f}%\")
"
open -a Skim "$DEST/$REMOTE" 2>/dev/null || true
