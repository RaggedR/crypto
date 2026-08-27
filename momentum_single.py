"""
Crypto momentum timing: single-asset directional strategy.

Uses Robin's bookmaker-pool momentum model to time entries and exits
on a single crypto asset. The idea: hold the asset when momentum
price > actual price (the volume flow supports a higher valuation).
Go to cash when momentum price < actual price.

No exchange accounts needed — all data from yfinance.

Usage:
  python momentum_single.py                         # BTC, defaults
  python momentum_single.py --asset ETH-USD         # ETH instead
  python momentum_single.py --lookback 90 --plot    # 90-day window + charts
  python momentum_single.py --start 2020-01-01      # custom period
"""

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Core algorithm (from Robin's bookmaker model) ────────────────────

def compute_rolling_momentum(history, lookback_days):
    """
    Compute rolling momentum price using the bookmaker pool model.

    Volume is normalised (divided by window mean) to prevent overflow
    on crypto's enormous trade volumes.

    Returns DataFrame with columns: Close, MomentumPrice, MomentumRatio.
    """
    closes = history["Close"].values
    volumes = history["Volume"].values
    dates = history.index

    momentum_prices = [np.nan] * len(history)
    momentum_ratios = [np.nan] * len(history)

    for i in range(lookback_days, len(history)):
        window_close = closes[i - lookback_days:i + 1]
        window_vol = volumes[i - lookback_days:i + 1]

        if len(window_close) < 10:
            continue

        # Daily returns
        returns = np.diff(window_close) / window_close[:-1]
        vols = window_vol[1:]

        # Normalise volume
        mean_vol = vols.mean()
        if mean_vol == 0:
            continue
        norm_vols = vols / mean_vol

        # Net flow = normalised_volume * return
        net_flows = norm_vols * returns

        # Run the bookmaker pool
        pool = float(window_close[0])
        mp = pool

        for j in range(len(net_flows)):
            pool += net_flows[j] * mp
            mp = pool

        momentum_prices[i] = mp
        momentum_ratios[i] = mp / closes[i]

    result = pd.DataFrame({
        "Close": closes,
        "MomentumPrice": momentum_prices,
        "MomentumRatio": momentum_ratios,
    }, index=dates)

    return result.dropna()


def simulate_timing(asset_symbol, lookback_days, start_date, end_date):
    """
    Simulate the timing strategy.

    Rule:
      - If MomentumRatio > 1.0: hold the asset (momentum supports price)
      - If MomentumRatio <= 1.0: hold cash (momentum says overvalued)

    Returns DataFrame with full simulation results.
    """
    fetch_start = start_date - timedelta(days=lookback_days + 60)

    print(f"Fetching {asset_symbol}...")
    ticker = yf.Ticker(asset_symbol)
    hist = ticker.history(start=fetch_start, end=end_date, interval="1d")
    print(f"  Got {len(hist)} days")

    if hist.empty:
        raise ValueError(f"No data for {asset_symbol}")

    print(f"Computing rolling momentum (lookback={lookback_days} days)...")
    df = compute_rolling_momentum(hist, lookback_days)

    # Trim to simulation period
    start_ts = pd.Timestamp(start_date)
    if df.index.tz is not None:
        start_ts = start_ts.tz_localize(df.index.tz)
    df = df[df.index >= start_ts]

    if df.empty:
        raise ValueError("No data in simulation period")

    # Signals: 1 = hold asset, 0 = hold cash
    df["signal"] = (df["MomentumRatio"] > 1.0).astype(int)

    # Position: use previous day's signal (can't trade on today's close)
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    # Returns
    df["asset_return"] = df["Close"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["asset_return"]

    # Cumulative
    df["cumulative_asset"] = (1 + df["asset_return"]).cumprod()
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cumulative_cash"] = 1.0  # cash benchmark (0% return)

    return df


def print_report(df, asset_symbol, lookback_days):
    """Print backtest summary."""
    total_days = len(df)
    days_in = (df["position"] == 1).sum()
    days_out = (df["position"] == 0).sum()

    total_return = (df["cumulative_strategy"].iloc[-1] - 1) * 100
    asset_return = (df["cumulative_asset"].iloc[-1] - 1) * 100

    # Trades: transitions between in and out
    transitions = (df["position"].diff().abs() > 0).sum()
    trades = transitions  # each transition is an entry or exit

    # Sharpe (annualised, 365 days for crypto)
    sr = df["strategy_return"]
    sharpe = sr.mean() / sr.std() * np.sqrt(365) if sr.std() > 0 else 0

    # Max drawdown
    cummax = df["cumulative_strategy"].cummax()
    drawdown = (df["cumulative_strategy"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Buy & hold max drawdown
    bh_cummax = df["cumulative_asset"].cummax()
    bh_drawdown = (df["cumulative_asset"] - bh_cummax) / bh_cummax
    bh_max_dd = bh_drawdown.min() * 100

    # Win rate of individual trades
    # A "trade" is a contiguous block of position=1
    in_blocks = []
    block_start = None
    for i in range(len(df)):
        if df["position"].iloc[i] == 1 and block_start is None:
            block_start = i
        elif df["position"].iloc[i] == 0 and block_start is not None:
            block_return = (df["Close"].iloc[i] / df["Close"].iloc[block_start] - 1) * 100
            in_blocks.append(block_return)
            block_start = None
    if block_start is not None:
        block_return = (df["Close"].iloc[-1] / df["Close"].iloc[block_start] - 1) * 100
        in_blocks.append(block_return)

    wins = sum(1 for r in in_blocks if r > 0)
    losses = sum(1 for r in in_blocks if r <= 0)
    avg_win = np.mean([r for r in in_blocks if r > 0]) if wins > 0 else 0
    avg_loss = np.mean([r for r in in_blocks if r <= 0]) if losses > 0 else 0

    # Periods where being out of market avoided a crash
    out_blocks = []
    block_start = None
    for i in range(len(df)):
        if df["position"].iloc[i] == 0 and block_start is None:
            block_start = i
        elif df["position"].iloc[i] == 1 and block_start is not None:
            avoided = (df["Close"].iloc[i] / df["Close"].iloc[block_start] - 1) * 100
            out_blocks.append(avoided)
            block_start = None
    if block_start is not None:
        avoided = (df["Close"].iloc[-1] / df["Close"].iloc[block_start] - 1) * 100
        out_blocks.append(avoided)

    crashes_avoided = sum(1 for r in out_blocks if r < -10)

    print()
    print("=" * 65)
    print(f"CRYPTO MOMENTUM TIMING: {asset_symbol}")
    print("=" * 65)
    print(f"  Period:             {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Lookback window:    {lookback_days} days")
    print(f"  Total trading days: {total_days}")
    print()
    print(f"  Days holding {asset_symbol[:3]}:  {days_in:5d} ({days_in/total_days*100:.1f}%)")
    print(f"  Days in cash:       {days_out:5d} ({days_out/total_days*100:.1f}%)")
    print(f"  Transitions:        {trades}")
    print(f"  Distinct trades:    {len(in_blocks)}")
    print()
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │ Strategy return:     {total_return:>+8.1f}%                  │")
    print(f"  │ Buy & hold return:   {asset_return:>+8.1f}%                  │")
    print(f"  │ Alpha:               {total_return - asset_return:>+8.1f}%                  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ Sharpe ratio:        {sharpe:>8.2f}                   │")
    print(f"  │ Max drawdown (strat):{max_dd:>+8.1f}%                  │")
    print(f"  │ Max drawdown (hold): {bh_max_dd:>+8.1f}%                  │")
    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(f"  Trade analysis ({len(in_blocks)} trades):")
    print(f"    Winners: {wins}  |  Losers: {losses}  |  Win rate: {wins/len(in_blocks)*100:.0f}%" if in_blocks else "    No trades")
    print(f"    Avg win:  {avg_win:+.1f}%  |  Avg loss: {avg_loss:+.1f}%")
    print(f"    Crashes avoided (>10% drop while out): {crashes_avoided}")
    print()

    return {
        "total_return": total_return,
        "asset_return": asset_return,
        "alpha": total_return - asset_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "bh_max_dd": bh_max_dd,
        "trades": len(in_blocks),
        "wins": wins,
        "losses": losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "days_in": days_in,
        "days_out": days_out,
        "total_days": total_days,
        "crashes_avoided": crashes_avoided,
    }


def plot_results(df, asset_symbol, lookback_days, stats, tag=""):
    """Generate charts."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Price and momentum price
    ax = axes[0]
    ax.plot(df.index, df["Close"], label=f"{asset_symbol} price",
            color="blue", alpha=0.8)
    ax.plot(df.index, df["MomentumPrice"], label="Momentum price",
            color="orange", alpha=0.8)
    # Shade when in position
    ymin, ymax = ax.get_ylim()
    for i in range(1, len(df)):
        if df["position"].iloc[i] == 1:
            ax.axvspan(df.index[i-1], df.index[i], alpha=0.05, color="green")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{asset_symbol}: Actual Price vs Momentum Price")
    ax.legend(loc="upper left", fontsize=9)

    # Plot 2: Momentum ratio + position
    ax = axes[1]
    ax.plot(df.index, df["MomentumRatio"], label="Momentum ratio",
            color="purple", alpha=0.8)
    ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--",
               label="Fair value (1.0)")
    ax.fill_between(df.index, df["MomentumRatio"], 1.0,
                    where=df["MomentumRatio"] > 1.0,
                    alpha=0.2, color="green", label="Hold asset")
    ax.fill_between(df.index, df["MomentumRatio"], 1.0,
                    where=df["MomentumRatio"] < 1.0,
                    alpha=0.2, color="red", label="Hold cash")
    ax.set_ylabel("Momentum Ratio")
    ax.set_title("Momentum Ratio (>1 = undervalued → hold | <1 = overvalued → cash)")
    ax.legend(loc="upper left", fontsize=9)

    # Plot 3: Cumulative returns
    ax = axes[2]
    ax.plot(df.index, df["cumulative_strategy"], label="Momentum timing",
            color="green", linewidth=2)
    ax.plot(df.index, df["cumulative_asset"], label=f"Buy & hold {asset_symbol}",
            color="blue", alpha=0.6)
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

    # Annotate final values
    strat_final = df["cumulative_strategy"].iloc[-1]
    hold_final = df["cumulative_asset"].iloc[-1]
    ax.annotate(f"Strategy: {strat_final:.2f}x",
                xy=(df.index[-1], strat_final),
                fontsize=9, fontweight="bold", color="green",
                ha="right", va="bottom")
    ax.annotate(f"Buy & hold: {hold_final:.2f}x",
                xy=(df.index[-1], hold_final),
                fontsize=9, fontweight="bold", color="blue",
                ha="right", va="top")

    ax.set_ylabel("Cumulative Return (1.0 = start)")
    ax.set_title("Cumulative Returns: Momentum Timing vs Buy & Hold")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"Momentum Timing Strategy: {asset_symbol} (lookback={lookback_days}d)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out = os.path.join(SCRIPT_DIR, f"timing_{asset_symbol.replace('-', '')}{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Crypto momentum timing backtest (single asset)"
    )
    parser.add_argument("--asset", default="BTC-USD",
                        help="Asset to trade. Default: BTC-USD")
    parser.add_argument("--lookback", type=int, default=180,
                        help="Momentum lookback window in days. Default: 180")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD). Default: 5 years ago")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--plot", action="store_true",
                        help="Generate charts")

    args = parser.parse_args()

    end_date = (datetime.strptime(args.end, "%Y-%m-%d")
                if args.end else datetime.now())
    start_date = (datetime.strptime(args.start, "%Y-%m-%d")
                  if args.start else end_date - timedelta(days=5*365))

    print()
    print(f"  Asset:     {args.asset}")
    print(f"  Lookback:  {args.lookback} days")
    print(f"  Period:    {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print()

    df = simulate_timing(
        asset_symbol=args.asset,
        lookback_days=args.lookback,
        start_date=start_date,
        end_date=end_date,
    )

    stats = print_report(df, args.asset, args.lookback)

    if args.plot:
        plot_results(df, args.asset, args.lookback, stats)

    csv_out = os.path.join(SCRIPT_DIR, f"timing_{args.asset.replace('-', '')}.csv")
    df.to_csv(csv_out)
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
