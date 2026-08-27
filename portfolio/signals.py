"""
Signal engine for the copper-gas-bitcoin-sol recycling portfolio.

Computes today's momentum signal for each of the four assets using
Robin's bookmaker-pool model (identical arithmetic to
momentum_single.py, so live signals match the backtest), then turns
the signals into target portfolio weights under the capital-recycling
rule with a hard cap on SOL.

Signals are computed on the *backtested* instrument (e.g. HG=F front-month
copper futures). Orders are placed in the *tradeable* instrument
(e.g. the CPER ETF). See TRADEABLE below.

Usage:
    python3 signals.py                # print today's signals + weights
    python3 signals.py --json         # machine-readable
    python3 signals.py --history 30   # last 30 days of signals
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# -- Portfolio definition ---------------------------------------------
# signal_symbol: what we compute momentum on (matches the backtest)
# lookback:      window in days (per-asset optimum from the study)
# cap:           maximum share of the portfolio, None = uncapped
# tradeable:     what we actually buy (venue-dependent, see oci.pdf)

ASSETS = {
    "COPPER": {"signal_symbol": "HG=F",    "lookback": 180, "cap": None,
               "tradeable": "CPER",    "venue": "IBKR"},
    "NATGAS": {"signal_symbol": "NG=F",    "lookback": 90,  "cap": None,
               "tradeable": "UNG",     "venue": "IBKR"},
    "BTC":    {"signal_symbol": "BTC-USD", "lookback": 180, "cap": None,
               "tradeable": "XBT/USD", "venue": "KRAKEN"},
    "SOL":    {"signal_symbol": "SOL-USD", "lookback": 180, "cap": 0.25,
               "tradeable": "SOL/USD", "venue": "KRAKEN"},
}

ORDER = ["COPPER", "NATGAS", "BTC", "SOL"]


# -- Core algorithm (unchanged from momentum_single.py) ----------------

def momentum_ratio(history, lookback_days):
    """
    Bookmaker-pool momentum ratio for the most recent day.

    Each day's net flow is (volume / mean volume) * return. Flows are
    compounded into a pool seeded at the price at the start of the
    window. The result is the 'momentum price'; we return its ratio to
    the actual close.

    Returns (ratio, close) or (None, close) if there is not enough data.
    """
    closes = history["Close"].values.astype(float)
    volumes = history["Volume"].values.astype(float)

    if len(closes) < lookback_days + 1:
        return None, (closes[-1] if len(closes) else None)

    window_close = closes[-(lookback_days + 1):]
    window_vol = volumes[-(lookback_days + 1):]

    returns = np.diff(window_close) / window_close[:-1]
    vols = window_vol[1:]

    mean_vol = vols.mean()
    if mean_vol == 0 or not np.isfinite(mean_vol):
        return None, closes[-1]
    norm_vols = vols / mean_vol

    net_flows = norm_vols * returns

    pool = float(window_close[0])
    mp = pool
    for flow in net_flows:
        pool += flow * mp
        mp = pool

    if not np.isfinite(mp):
        return None, closes[-1]

    return mp / closes[-1], closes[-1]


# -- Allocation -------------------------------------------------------

def target_weights(signals):
    """
    Capital-recycling weights with per-asset caps.

    Equal weight across every asset whose signal is on. Any weight above
    an asset's cap is redistributed to the other active, uncapped assets;
    if there are none, it stays in cash.

    signals: {asset: bool}
    Returns {asset: weight}; weights sum to <= 1, remainder is cash.
    """
    active = [a for a in ORDER if signals.get(a)]
    if not active:
        return {a: 0.0 for a in ORDER}

    w = {a: (1.0 / len(active) if a in active else 0.0) for a in ORDER}

    # Apply caps, spilling the excess onto uncapped active assets.
    for _ in range(len(active)):          # iterate to a fixed point
        excess = 0.0
        for a in active:
            cap = ASSETS[a]["cap"]
            if cap is not None and w[a] > cap + 1e-12:
                excess += w[a] - cap
                w[a] = cap
        if excess <= 1e-12:
            break
        receivers = [a for a in active if ASSETS[a]["cap"] is None]
        if not receivers:
            break                        # excess stays in cash
        for a in receivers:
            w[a] += excess / len(receivers)

    return w


def fetch_signals(asof=None):
    """Fetch data and compute signals for all four assets."""
    end = (asof or datetime.now(timezone.utc)) + timedelta(days=1)
    out = {}
    for name in ORDER:
        cfg = ASSETS[name]
        lb = cfg["lookback"]
        start = end - timedelta(days=lb + 200)   # slack for weekends/holidays
        try:
            hist = yf.Ticker(cfg["signal_symbol"]).history(
                start=start.date(), end=end.date(), interval="1d",
                auto_adjust=False,
            )
            hist = hist.dropna(subset=["Close", "Volume"])
            ratio, close = momentum_ratio(hist, lb)
            out[name] = {
                "symbol": cfg["signal_symbol"],
                "lookback": lb,
                "close": None if close is None else float(close),
                "ratio": None if ratio is None else float(ratio),
                "signal": bool(ratio is not None and ratio > 1.0),
                "asof": (str(hist.index[-1].date()) if len(hist) else None),
                "cap": cfg["cap"],
                "tradeable": cfg["tradeable"],
                "venue": cfg["venue"],
                "error": None,
            }
        except Exception as exc:                       # noqa: BLE001
            out[name] = {
                "symbol": cfg["signal_symbol"], "lookback": lb,
                "close": None, "ratio": None, "signal": False, "asof": None,
                "cap": cfg["cap"], "tradeable": cfg["tradeable"],
                "venue": cfg["venue"], "error": str(exc),
            }
    return out


def build_state(asof=None):
    """Full state dict: signals, weights, metadata."""
    sig = fetch_signals(asof)
    weights = target_weights({k: v["signal"] for k, v in sig.items()})
    cash = max(0.0, 1.0 - sum(weights.values()))
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assets": sig,
        "weights": weights,
        "cash": cash,
        "n_active": sum(1 for v in sig.values() if v["signal"]),
        "errors": [k for k, v in sig.items() if v["error"]],
    }


def render_text(state):
    lines = []
    lines.append(f"Signals as of {state['generated']}  "
                 f"({state['n_active']}/4 active)")
    lines.append("")
    lines.append(f"{'Asset':8s} {'Symbol':9s} {'Close':>12s} {'Ratio':>8s} "
                 f"{'Signal':>7s} {'Weight':>8s}  Trade as")
    lines.append("-" * 76)
    for a in ORDER:
        v = state["assets"][a]
        close = "n/a" if v["close"] is None else f"{v['close']:,.2f}"
        ratio = "n/a" if v["ratio"] is None else f"{v['ratio']:.3f}"
        sig = "HOLD" if v["signal"] else "cash"
        cap = " (capped)" if (v["cap"] is not None
                              and state["weights"][a] >= v["cap"] - 1e-9
                              and v["signal"]) else ""
        lines.append(f"{a:8s} {v['symbol']:9s} {close:>12s} {ratio:>8s} "
                     f"{sig:>7s} {state['weights'][a]*100:7.1f}%  "
                     f"{v['tradeable']} @ {v['venue']}{cap}")
    lines.append("-" * 76)
    lines.append(f"{'CASH':8s} {'':9s} {'':>12s} {'':>8s} {'':>7s} "
                 f"{state['cash']*100:7.1f}%")
    if state["errors"]:
        lines.append("")
        lines.append(f"WARNING: no data for {', '.join(state['errors'])} "
                     f"- treated as 'cash' (fail-safe).")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--out", default=None, help="write JSON state to this path")
    args = p.parse_args()

    state = build_state()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(state, fh, indent=2)

    print(json.dumps(state, indent=2) if args.json else render_text(state))
    return 1 if len(state["errors"]) == len(ORDER) else 0


if __name__ == "__main__":
    sys.exit(main())
