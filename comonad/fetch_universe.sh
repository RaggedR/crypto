#!/usr/bin/env bash
# Fetch a universe of liquid Binance USDT daily candles for the
# train/test forward-model study. Deterministic list -> reproducible.
set -euo pipefail
cd "$(dirname "$0")"
PY=../venv/bin/python

COINS=(BTC ETH BNB XRP LTC ADA EOS TRX XLM ETC BCH NEO IOTA QTUM ONT \
       VET LINK ATOM ALGO MATIC DOGE DOT SOL AVAX UNI FIL AAVE SAND MANA FTM)

for c in "${COINS[@]}"; do
  if [ -f "data/${c}USDT.csv" ]; then
    echo "skip ${c}USDT (cached)"
  else
    $PY fetch.py --symbol "${c}/USDT" --since 2017-08-01 || echo "FAILED ${c}"
  fi
done
echo "universe fetch done: $(ls data/*.csv | wc -l) files"
