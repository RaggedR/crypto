"""
CCXT -> CSV bridge for the tape-comonad backtester.

Pulls daily OHLCV candles from a liquid exchange (Binance by default,
Kraken as fallback) where the *volume* tape is real -- the bookmaker
model weights each day's return by volume, so a representative volume
series is not optional. Writes a plain CSV the Haskell side reads:

    date,open,high,low,close,volume

Usage:
  python fetch.py                         # BTC/USDT daily from Binance
  python fetch.py --symbol ETH/USDT
  python fetch.py --exchange kraken --symbol XBT/USD
  python fetch.py --timeframe 1d --since 2017-01-01
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import ccxt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def iso_to_ms(s):
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_all(exchange, symbol, timeframe, since_ms):
    """Page through fetch_ohlcv until we reach 'now'.

    One call returns one page (exchange-capped, ~500-1000 candles), so
    we loop, advancing `since` past the last candle each time.
    """
    all_rows = []
    limit = 1000
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        # drop the overlap candle the next page would repeat
        if all_rows and batch[0][0] == all_rows[-1][0]:
            batch = batch[1:]
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + tf_ms
        if since_ms > exchange.milliseconds():
            break
        time.sleep((exchange.rateLimit or 200) / 1000.0)
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--since", default="2017-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    ex_class = getattr(ccxt, args.exchange, None)
    if ex_class is None:
        sys.exit(f"unknown exchange: {args.exchange}")
    exchange = ex_class({"enableRateLimit": True})

    print(f"fetching {args.symbol} {args.timeframe} from {args.exchange} "
          f"since {args.since} ...", file=sys.stderr)
    rows = fetch_all(exchange, args.symbol, args.timeframe, iso_to_ms(args.since))
    if not rows:
        sys.exit("no data returned")

    name = args.out or (args.symbol.replace("/", "") + ".csv")
    out_path = name if os.path.isabs(name) else os.path.join(DATA_DIR, name)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for ts, o, h, l, c, v in rows:
            d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            w.writerow([d, o, h, l, c, v])

    print(f"wrote {len(rows)} candles to {out_path}", file=sys.stderr)
    print(out_path)


if __name__ == "__main__":
    main()
