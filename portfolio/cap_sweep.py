"""
Choose the Solana cap for the copper-gas-bitcoin-sol recycling portfolio.

Regenerates rolling momentum for all four assets (the saved timing_*.csv
files in the parent directory are stale and inconsistent with the study),
builds the capital-recycling portfolio for a range of SOL caps, and
reports the risk/return trade-off for each.

Calendar handling: crypto trades 7 days a week, futures do not. We use
the crypto calendar as the master, forward-fill futures closes onto
non-trading days (so a held position persists over the weekend) and
assign those days zero return. Signals likewise persist.

Usage:
    python3 cap_sweep.py                    # full sweep
    python3 cap_sweep.py --refresh          # re-download prices
"""

import argparse
import os

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

SPEC = {
    "COPPER": ("HG=F", 180),
    "NATGAS": ("NG=F", 90),
    "BTC": ("BTC-USD", 180),
    "SOL": ("SOL-USD", 180),
}
ORDER = ["COPPER", "NATGAS", "BTC", "SOL"]
START = "2021-04-19"          # first date all four have a full lookback
CAPITAL = 10_000.0


def fetch(symbol, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, symbol.replace("=", "_").replace("-", "_") + ".csv")
    if os.path.exists(path) and not refresh:
        d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        return d
    h = yf.Ticker(symbol).history(start="2020-01-01", interval="1d",
                                  auto_adjust=False)
    h = h.dropna(subset=["Close", "Volume"])[["Close", "Volume"]]
    h.index = pd.to_datetime(h.index, utc=True).tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")]
    h.index.name = "Date"
    h.to_csv(path)
    return h


def rolling_ratio(hist, lookback):
    """Bookmaker-pool momentum ratio for each day (native calendar)."""
    closes = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)
    out = np.full(len(closes), np.nan)
    for i in range(lookback, len(closes)):
        wc = closes[i - lookback:i + 1]
        wv = volumes[i - lookback:i + 1]
        rets = np.diff(wc) / wc[:-1]
        vols = wv[1:]
        mv = vols.mean()
        if mv == 0 or not np.isfinite(mv):
            continue
        flows = (vols / mv) * rets
        pool = float(wc[0])
        mp = pool
        for f in flows:
            pool += f * mp
            mp = pool
        if np.isfinite(mp) and closes[i] != 0:
            out[i] = mp / closes[i]
    return pd.Series(out, index=hist.index)


def build_panel(refresh=False):
    """Master calendar, per-asset returns and lagged signals."""
    hist, ratio = {}, {}
    for name, (sym, lb) in SPEC.items():
        h = fetch(sym, refresh)
        hist[name] = h
        ratio[name] = rolling_ratio(h, lb)

    cal = hist["BTC"].index                     # 7-day crypto calendar
    cal = cal[cal >= pd.Timestamp(START)]

    rets, sigs = {}, {}
    for name in ORDER:
        close = hist[name]["Close"].reindex(cal).ffill()
        rets[name] = close.pct_change().fillna(0.0)
        s = (ratio[name] > 1.0).where(ratio[name].notna())
        s = s.reindex(cal).ffill().fillna(0.0).astype(bool)
        sigs[name] = s.shift(1).fillna(False).astype(bool)   # trade on yesterday

    return pd.DataFrame(rets)[ORDER], pd.DataFrame(sigs)[ORDER]


def weights_for(sig_row, cap):
    """Equal weight across active assets; SOL capped, excess to others."""
    active = [a for a in ORDER if sig_row[a]]
    w = {a: 0.0 for a in ORDER}
    if not active:
        return w
    for a in active:
        w[a] = 1.0 / len(active)
    if cap is not None and w["SOL"] > cap:
        excess = w["SOL"] - cap
        w["SOL"] = cap
        others = [a for a in active if a != "SOL"]
        if others:
            for a in others:
                w[a] += excess / len(others)
    return w


def run(P, S, cap):
    W = pd.DataFrame(
        [weights_for(S.loc[d], cap) for d in S.index],
        index=S.index, columns=ORDER)
    port = (W * P).sum(axis=1)
    nav = CAPITAL * (1 + port).cumprod()
    eq = nav / CAPITAL
    dd = nav / nav.cummax() - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    r = port
    downside = r[r < 0].std()
    return {
        "cap": cap,
        "final": nav.iloc[-1],
        "total": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(365),
        "sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
        "sortino": (r.mean() / downside * np.sqrt(365)
                    if downside and downside > 0 else np.nan),
        "maxdd": dd.min(),
        "calmar": ((eq.iloc[-1] ** (1 / years) - 1) / abs(dd.min())
                   if dd.min() < 0 else np.nan),
        "sol_avg_w": W["SOL"].mean(),
        "sol_share_of_risk": None,
        "worst_day": r.min(),
        "nav": nav,
        "W": W,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    P, S = build_panel(args.refresh)

    print(f"Period: {S.index[0].date()} to {S.index[-1].date()}  "
          f"({len(S)} days, {(S.index[-1]-S.index[0]).days/365.25:.2f} years)")
    print(f"Start capital: ${CAPITAL:,.0f}\n")

    print("Signal activity (fraction of days each asset says HOLD):")
    for a in ORDER:
        print(f"  {a:7s} {S[a].mean()*100:5.1f}%")
    n_active = S.sum(axis=1)
    print("\nHow many assets active at once:")
    for k in range(5):
        print(f"  {k} active: {(n_active==k).mean()*100:5.1f}%")
    print(f"  capital deployed: {(n_active>0).mean()*100:.1f}% of days")
    print("\nOverlap: fraction of SOL-active days where SOL is the ONLY signal:")
    solonly = ((S["SOL"]) & (n_active == 1)).sum() / max(1, S["SOL"].sum())
    print(f"  {solonly*100:.1f}%   <- these are the days an uncapped rule "
          f"puts 100% in SOL")

    caps = [0.0, 0.10, 0.15, 0.20, 0.25, 0.3333, 0.50, None]
    results = [run(P, S, c) for c in caps]

    print("\n" + "=" * 104)
    print(f"{'Cap':>7s} {'Final $':>12s} {'CAGR':>8s} {'Vol':>7s} "
          f"{'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>7s} "
          f"{'AvgSOLw':>8s} {'WorstDay':>9s}")
    print("=" * 104)
    for r in results:
        lab = "none" if r["cap"] is None else f"{r['cap']*100:.0f}%"
        print(f"{lab:>7s} {r['final']:>12,.0f} {r['cagr']*100:>7.1f}% "
              f"{r['vol']*100:>6.1f}% {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['maxdd']*100:>7.1f}% {r['calmar']:>7.2f} "
              f"{r['sol_avg_w']*100:>7.1f}% {r['worst_day']*100:>8.1f}%")
    print("=" * 104)

    # Marginal analysis: what each step up in the cap buys and costs.
    print("\nMarginal trade-off, relative to the uncapped rule:")
    base = [r for r in results if r["cap"] is None][0]
    print(f"{'Cap':>7s} {'% of uncapped return kept':>26s} "
          f"{'% of uncapped drawdown taken':>29s} {'return kept per unit risk':>26s}")
    for r in results:
        if r["cap"] is None or r["cap"] == 0.0:
            continue
        kept = r["total"] / base["total"]
        pain = r["maxdd"] / base["maxdd"]
        print(f"{r['cap']*100:>6.0f}% {kept*100:>25.1f}% {pain*100:>28.1f}% "
              f"{kept/pain:>25.2f}")

    os.makedirs(CACHE, exist_ok=True)
    out = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("nav", "W")} for r in results])
    out.to_csv(os.path.join(CACHE, "cap_sweep.csv"), index=False)
    pd.DataFrame({("none" if r["cap"] is None else f"{r['cap']:.4f}"): r["nav"]
                  for r in results}).to_csv(os.path.join(CACHE, "cap_navs.csv"))
    print(f"\nWrote {CACHE}/cap_sweep.csv and cap_navs.csv")


if __name__ == "__main__":
    main()
