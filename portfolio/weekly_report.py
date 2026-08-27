"""
Weekly PDF report for the copper-gas-bitcoin-sol recycling portfolio.

Rebuilds the whole paper-trading equity curve from scratch on every run,
using the same bookmaker-pool arithmetic as the backtest. That makes the
report stateless and self-healing: a missed week, a crashed run, or a
corrupted file cannot cause the tracked NAV to drift, because nothing is
carried forward.

If a broker snapshot exists (accounts.json, written by the execution
layer once live trading starts), the real NAV is reported alongside the
paper NAV so the two can be compared.

Outputs, all under --outdir:
    portfolio-YYYY-Www.pdf   this week's report
    latest.pdf               copy of the most recent report
    latest.json             machine-readable summary (for Claude)
    nav_history.csv         daily paper NAV series

Usage:
    python3 weekly_report.py                      # default paths
    python3 weekly_report.py --capital 10000
    python3 weekly_report.py --outdir ~/portfolio/reports
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from signals import ASSETS, ORDER, target_weights, build_state

INCEPTION = "2026-08-01"     # when paper tracking starts
ACCENT = "#1e4678"
GREEN = "#287828"
RED = "#b42828"
GREY = "#888888"
PALETTE = {"COPPER": "#b87333", "NATGAS": "#4c9be8",
           "BTC": "#f2a900", "SOL": "#9945ff"}


# -- Paper equity curve -----------------------------------------------

def rolling_ratios(hist, lookback):
    """Momentum ratio for every day with a full lookback window."""
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


def build_paper_curve(capital, inception=INCEPTION):
    """
    Daily paper NAV since inception.

    Signals use the previous day's ratio (you cannot trade on a close you
    have not seen yet) - the same one-day lag as the backtest.
    """
    start = pd.Timestamp(inception)
    fetch_from = (start - timedelta(days=400)).date()
    end = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    ratios, rets, closes = {}, {}, {}
    for name in ORDER:
        cfg = ASSETS[name]
        h = yf.Ticker(cfg["signal_symbol"]).history(
            start=fetch_from, end=end, interval="1d", auto_adjust=False)
        h = h.dropna(subset=["Close", "Volume"])
        if h.empty:
            continue
        h.index = pd.to_datetime(h.index, utc=True).tz_localize(None).normalize()
        h = h[~h.index.duplicated(keep="last")]
        ratios[name] = rolling_ratios(h, cfg["lookback"])
        rets[name] = h["Close"].pct_change()
        closes[name] = h["Close"]

    if not ratios:
        return None, None, None

    R = pd.DataFrame(ratios).sort_index()
    P = pd.DataFrame(rets).reindex(R.index)

    # Trade on yesterday's signal.
    S = (R > 1.0) & R.notna()
    S = S.shift(1)
    S = S.where(S.notna(), False).astype(bool)

    idx = R.index[R.index >= start]
    if len(idx) == 0:
        return None, None, None
    S, P = S.reindex(idx).fillna(False), P.reindex(idx).fillna(0.0)

    W = pd.DataFrame(0.0, index=idx, columns=ORDER)
    for d in idx:
        w = target_weights({a: bool(S.get(a, pd.Series(dtype=bool)).get(d, False))
                            for a in ORDER})
        for a in ORDER:
            W.at[d, a] = w[a]

    port = (W * P.reindex(columns=ORDER).fillna(0.0)).sum(axis=1)
    nav = capital * (1 + port).cumprod()
    return nav, W, closes


def stats(nav):
    if nav is None or len(nav) < 2:
        return {}
    r = nav.pct_change().dropna()
    eq = nav / nav.iloc[0]
    dd = (nav / nav.cummax() - 1)
    week_ago = nav.index[-1] - timedelta(days=7)
    prior = nav[nav.index <= week_ago]
    wk = (nav.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else np.nan
    days = max(1, (nav.index[-1] - nav.index[0]).days)
    return {
        "nav": float(nav.iloc[-1]),
        "start_nav": float(nav.iloc[0]),
        "total_return": float(eq.iloc[-1] - 1),
        "week_return": float(wk) if wk == wk else None,
        "annualised": float(eq.iloc[-1] ** (365.0 / days) - 1),
        "sharpe": (float(r.mean() / r.std() * np.sqrt(252))
                   if r.std() > 0 else None),
        "max_dd": float(dd.min()),
        "current_dd": float(dd.iloc[-1]),
        "days": int(days),
        "deployed_pct": None,
    }


# -- PDF rendering ----------------------------------------------------

def _txt(ax, x, y, s, size=10, weight="normal", color="black", family=None,
         ha="left", va="top"):
    ax.text(x, y, s, size=size, weight=weight, color=color, ha=ha, va=va,
            family=family, transform=ax.transAxes)


def render_pdf(path, state, st, nav, W, broker):
    week = datetime.now(timezone.utc).strftime("%G-W%V")
    with PdfPages(path) as pdf:

        # -- Page 1: summary ------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.06)

        head = fig.add_axes([0.08, 0.83, 0.86, 0.12]); head.axis("off")
        _txt(head, 0, 1.0, "Recycling Portfolio", 22, "bold", ACCENT)
        _txt(head, 0, 0.55, "Copper  |  Natural Gas  |  Bitcoin  |  Solana",
             12, color=GREY)
        _txt(head, 0, 0.22,
             f"Week {week}   |   generated "
             f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", 9, color=GREY)

        # Headline tiles
        tiles = fig.add_axes([0.08, 0.70, 0.86, 0.11]); tiles.axis("off")
        if st:
            wk = st.get("week_return")
            items = [
                ("Paper NAV", f"${st['nav']:,.0f}", ACCENT),
                ("This week", "n/a" if wk is None else f"{wk*100:+.2f}%",
                 GREEN if (wk or 0) >= 0 else RED),
                ("Since inception", f"{st['total_return']*100:+.2f}%",
                 GREEN if st["total_return"] >= 0 else RED),
                ("Max drawdown", f"{st['max_dd']*100:.1f}%", RED),
            ]
            for i, (lab, val, col) in enumerate(items):
                x = i * 0.25
                tiles.add_patch(plt.Rectangle((x + 0.005, 0.05), 0.24, 0.9,
                                              transform=tiles.transAxes,
                                              facecolor="#f2f5fa",
                                              edgecolor="#d5dee9"))
                _txt(tiles, x + 0.02, 0.80, lab, 8, color=GREY)
                _txt(tiles, x + 0.02, 0.52, val, 15, "bold", col)
        else:
            _txt(tiles, 0, 0.6, "No price history available this run.", 11, color=RED)

        # Signal table
        tab = fig.add_axes([0.08, 0.44, 0.86, 0.24]); tab.axis("off")
        _txt(tab, 0, 1.0, "Today's signals and target allocation", 12, "bold", ACCENT)
        rows = [["Asset", "Signal on", "Ratio", "Last close", "Target", "Instrument"]]
        for a in ORDER:
            v = state["assets"][a]
            rows.append([
                a.title(),
                "HOLD" if v["signal"] else "cash",
                "n/a" if v["ratio"] is None else f"{v['ratio']:.3f}",
                "n/a" if v["close"] is None else f"{v['close']:,.2f}",
                f"{state['weights'][a]*100:.1f}%"
                + ("  (cap)" if (v["cap"] is not None and v["signal"]
                                and state["weights"][a] >= v["cap"] - 1e-9) else ""),
                f"{v['tradeable']} | {v['venue'].title()}",
            ])
        rows.append(["Cash", "", "", "", f"{state['cash']*100:.1f}%", ""])

        t = tab.table(cellText=rows[1:], colLabels=rows[0],
                      cellLoc="left", colLoc="left",
                      colWidths=[0.14, 0.13, 0.11, 0.17, 0.19, 0.26],
                      bbox=[0, 0.05, 1, 0.85])
        t.auto_set_font_size(False); t.set_fontsize(8.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if r == 0:
                cell.set_facecolor(ACCENT)
                cell.set_text_props(color="white", weight="bold")
            elif r == len(rows) - 1:
                cell.set_facecolor("#f5f5f5")
                cell.set_text_props(style="italic")
            elif rows[r][1] == "HOLD":
                cell.set_facecolor("#eef7ee")

        # Allocation bar
        bar = fig.add_axes([0.08, 0.27, 0.86, 0.13])
        left = 0.0
        for a in ORDER:
            w = state["weights"][a]
            if w <= 0:
                continue
            bar.barh(0, w, left=left, color=PALETTE[a], edgecolor="white")
            if w > 0.06:
                bar.text(left + w / 2, 0, f"{a}\n{w*100:.0f}%", ha="center",
                         va="center", size=8.5, color="white", weight="bold")
            left += w
        if state["cash"] > 0.005:
            bar.barh(0, state["cash"], left=left, color="#cccccc",
                     edgecolor="white")
            if state["cash"] > 0.06:
                bar.text(left + state["cash"] / 2, 0,
                         f"CASH\n{state['cash']*100:.0f}%", ha="center",
                         va="center", size=8.5, color="#444444", weight="bold")
        bar.set_xlim(0, 1); bar.set_ylim(-0.5, 0.5)
        bar.set_yticks([]); bar.set_xticks([0, .25, .5, .75, 1])
        bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], size=8)
        bar.set_title("Where the money should be right now", size=10,
                      loc="left", color=ACCENT, weight="bold")
        for s in bar.spines.values():
            s.set_visible(False)

        # Notes / reconciliation
        note = fig.add_axes([0.08, 0.05, 0.86, 0.19]); note.axis("off")
        y = 1.0
        _txt(note, 0, y, "Notes", 11, "bold", ACCENT); y -= 0.14
        msgs = []
        if broker:
            real = broker.get("total_value")
            if real is not None and st:
                diff = real / st["nav"] - 1
                msgs.append(f"Broker NAV ${real:,.2f} vs paper ${st['nav']:,.2f} "
                            f"({diff*100:+.2f}% tracking difference).")
            msgs.append(f"Broker snapshot taken {broker.get('asof', 'unknown')}.")
        else:
            msgs.append("PAPER MODE - no broker snapshot found. Figures below are "
                        "simulated, not realised. No money is at risk.")
        if state["errors"]:
            msgs.append(f"No price data for {', '.join(state['errors'])}; treated "
                        f"as 'cash' (fail-safe). Investigate before trading.")
        msgs.append("Paper NAV assumes execution at the daily close with no fees, "
                    "spread, slippage or tax. Live results will be lower.")
        msgs.append(f"Solana is capped at "
                    f"{ASSETS['SOL']['cap']*100:.0f}% of the portfolio by design.")
        for m in msgs:
            for line in _wrap(m, 105):
                _txt(note, 0, y, line, 8.5); y -= 0.085

        pdf.savefig(fig); plt.close(fig)

        # -- Page 2: curves -------------------------------------------
        if nav is not None and len(nav) > 1:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.subplots_adjust(left=0.10, right=0.94, top=0.94, bottom=0.06,
                               hspace=0.35)

            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(nav.index, nav.values, color=ACCENT, lw=1.6)
            ax1.fill_between(nav.index, st["start_nav"], nav.values,
                             where=(nav.values >= st["start_nav"]),
                             color=GREEN, alpha=0.12)
            ax1.fill_between(nav.index, st["start_nav"], nav.values,
                             where=(nav.values < st["start_nav"]),
                             color=RED, alpha=0.12)
            ax1.axhline(st["start_nav"], color=GREY, ls="--", lw=0.8)
            ax1.set_title("Paper NAV since inception", size=11, loc="left",
                          color=ACCENT, weight="bold")
            ax1.set_ylabel("USD")
            ax1.grid(alpha=0.25)

            ax2 = fig.add_subplot(3, 1, 2)
            dd = (nav / nav.cummax() - 1) * 100
            ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.3)
            ax2.plot(dd.index, dd.values, color=RED, lw=1.0)
            ax2.set_title("Drawdown from peak", size=11, loc="left",
                          color=ACCENT, weight="bold")
            ax2.set_ylabel("%")
            ax2.grid(alpha=0.25)

            ax3 = fig.add_subplot(3, 1, 3)
            for i, a in enumerate(ORDER):
                on = W[a] > 0
                ax3.fill_between(W.index, i - 0.4, i + 0.4, where=on,
                                 color=PALETTE[a], alpha=0.85,
                                 step="post", linewidth=0)
            ax3.set_yticks(range(len(ORDER)))
            ax3.set_yticklabels([a.title() for a in ORDER], size=9)
            ax3.set_ylim(-0.6, len(ORDER) - 0.4)
            ax3.set_title("When each asset was held (shaded = in position)",
                          size=11, loc="left", color=ACCENT, weight="bold")
            ax3.grid(alpha=0.2, axis="x")

            pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = f"Recycling Portfolio Report {week}"
        d["Author"] = "signals.py / weekly_report.py"


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# -- Main -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Starting paper capital. Default: 10000")
    p.add_argument("--outdir", default=os.path.expanduser("~/portfolio/reports"))
    p.add_argument("--accounts", default=os.path.expanduser("~/portfolio/accounts.json"),
                   help="Optional broker snapshot written by the execution layer")
    p.add_argument("--inception", default=INCEPTION)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    state = build_state()
    nav, W, _ = build_paper_curve(args.capital, args.inception)
    st = stats(nav)
    if st and W is not None:
        st["deployed_pct"] = float((W.sum(axis=1) > 0).mean())

    broker = None
    if os.path.exists(args.accounts):
        try:
            with open(args.accounts) as fh:
                broker = json.load(fh)
        except Exception:                              # noqa: BLE001
            broker = None

    week = datetime.now(timezone.utc).strftime("%G-W%V")
    out = os.path.join(args.outdir, f"portfolio-{week}.pdf")
    render_pdf(out, state, st, nav, W, broker)
    shutil.copyfile(out, os.path.join(args.outdir, "latest.pdf"))

    if nav is not None:
        nav.rename("nav").to_csv(os.path.join(args.outdir, "nav_history.csv"),
                                 index_label="date")

    summary = {"week": week, "report": out, "stats": st,
               "signals": {a: state["assets"][a]["signal"] for a in ORDER},
               "ratios": {a: state["assets"][a]["ratio"] for a in ORDER},
               "weights": state["weights"], "cash": state["cash"],
               "mode": "live" if broker else "paper",
               "errors": state["errors"],
               "generated": state["generated"]}
    with open(os.path.join(args.outdir, "latest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {out}")
    if st:
        print(f"  Paper NAV ${st['nav']:,.2f}  "
              f"week {(st['week_return'] or 0)*100:+.2f}%  "
              f"total {st['total_return']*100:+.2f}%  "
              f"maxDD {st['max_dd']*100:.1f}%")
    print(f"  Mode: {summary['mode']}   Active: {state['n_active']}/4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
