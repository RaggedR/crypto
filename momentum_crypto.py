"""
Crypto momentum pairs trading: BTC/ETH relative value strategy.

Adapts Robin's bookmaker-pool momentum model from ~/git/momentum/ to
cryptocurrency pairs. The idea: when ETH is under-momentum-priced and
BTC is over-momentum-priced, rotate from BTC into ETH. Rotate back
when the spread recalibrates.

This is the simulation/backtest tool. No exchange accounts needed —
all data comes from yfinance (free, public, no auth).

Usage:
  python momentum_crypto.py                    # backtest with defaults
  python momentum_crypto.py --pair ETH-BTC     # specific pair
  python momentum_crypto.py --lookback 365     # 1 year momentum history
  python momentum_crypto.py --threshold 0.02   # 2% spread to trigger
  python momentum_crypto.py --plot             # generate charts
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Core algorithm (from Robin's bookmaker model) ────────────────────

def compute_momentum_price(history):
    """
    Bookmaker model adapted for crypto.

    For crypto, volumes are enormous (millions of coins traded daily),
    so we normalise volume by dividing by the mean volume over the
    window. This preserves the relative signal — days with above-average
    volume push the pool harder — while keeping numbers in a stable
    range that won't overflow.

    The pool starts at the first closing price. Each day, the pool
    adjusts by: normalised_volume * daily_return * momentum_price.
    This is faithful to Robin's original model: volume * return gives
    net directional flow, and we scale by momentum_price to keep the
    pool in price-space.

    Args:
        history: DataFrame with 'Close' and 'Volume' columns.

    Returns:
        Series of momentum prices, same index as input (minus first row).
    """
    if len(history) < 10:
        return None

    h = history.copy()
    h["DailyReturn"] = h["Close"].pct_change()

    # Normalise volume to prevent overflow: express each day's volume
    # as a fraction of mean volume. A "1.0" day has average volume;
    # a "2.0" day has twice the average. This keeps net_flow manageable.
    mean_vol = h["Volume"].mean()
    if mean_vol == 0:
        return None
    h["NormVolume"] = h["Volume"] / mean_vol
    h["NetVolumeFlow"] = h["NormVolume"] * h["DailyReturn"]
    h = h.dropna()

    if len(h) < 10:
        return None

    pool = float(h["Close"].iloc[0])
    mp = pool

    momentum_prices = [mp]

    for i in range(1, len(h)):
        net_flow = float(h["NetVolumeFlow"].iloc[i])
        pool += net_flow * mp
        mp = pool
        momentum_prices.append(mp)

    return pd.Series(momentum_prices, index=h.index, name="MomentumPrice")


def compute_momentum_ratio(history):
    """
    Compute momentum_price / actual_price over time.

    > 1.0 means undervalued (momentum supports a higher price)
    < 1.0 means overvalued (momentum supports a lower price)
    """
    mp = compute_momentum_price(history)
    if mp is None:
        return None

    h = history.loc[mp.index].copy()
    ratio = mp / h["Close"]
    ratio.name = "MomentumRatio"
    return ratio


# ── Data fetching ────────────────────────────────────────────────────

def fetch_crypto(symbol, start, end):
    """Download daily OHLCV from yfinance for a crypto asset."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, interval="1d")
    if hist.empty:
        raise ValueError(f"No data for {symbol} between {start} and {end}")
    return hist


# ── Pairs trading simulation ─────────────────────────────────────────

def simulate_pairs_trade(base_symbol, quote_symbol, lookback_days,
                         threshold, start_date, end_date):
    """
    Simulate the relative value strategy.

    We compute momentum ratios for both assets using a rolling window.
    When base is undervalued (ratio > 1+threshold) AND quote is
    overvalued (ratio < 1-threshold), we go long base / short quote.
    We close when the spread normalises.

    For the BTC/ETH case:
        base  = ETH-USD  (we buy this when it's cheap)
        quote = BTC-USD  (we sell this when it's expensive)

    Returns a DataFrame with signals, positions, and cumulative returns.
    """
    # Fetch data with extra lookback for the momentum calculation
    fetch_start = start_date - timedelta(days=lookback_days + 30)

    print(f"Fetching {base_symbol}...")
    base_hist = fetch_crypto(base_symbol, fetch_start, end_date)
    print(f"  Got {len(base_hist)} days")

    print(f"Fetching {quote_symbol}...")
    quote_hist = fetch_crypto(quote_symbol, fetch_start, end_date)
    print(f"  Got {len(quote_hist)} days")

    # Align dates
    common_idx = base_hist.index.intersection(quote_hist.index)
    base_hist = base_hist.loc[common_idx]
    quote_hist = quote_hist.loc[common_idx]

    print(f"Common trading days: {len(common_idx)}")
    print()

    # Compute rolling momentum ratios
    print("Computing rolling momentum ratios...")
    results = []

    sim_start_idx = None
    for i, date in enumerate(common_idx):
        if date.tz_localize(None) if date.tzinfo else date >= start_date:
            if sim_start_idx is None:
                sim_start_idx = i
            break

    if sim_start_idx is None or sim_start_idx < lookback_days:
        # Find the first valid simulation point
        sim_start_idx = lookback_days

    for i in range(sim_start_idx, len(common_idx)):
        window_start = max(0, i - lookback_days)
        date = common_idx[i]

        base_window = base_hist.iloc[window_start:i+1]
        quote_window = quote_hist.iloc[window_start:i+1]

        base_ratio = compute_momentum_ratio(base_window)
        quote_ratio = compute_momentum_ratio(quote_window)

        if base_ratio is None or quote_ratio is None:
            continue

        results.append({
            "date": date,
            "base_close": base_hist["Close"].iloc[i],
            "quote_close": quote_hist["Close"].iloc[i],
            "base_momentum_ratio": base_ratio.iloc[-1],
            "quote_momentum_ratio": quote_ratio.iloc[-1],
            "spread": base_ratio.iloc[-1] - quote_ratio.iloc[-1],
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("Not enough data for simulation.")
        return df

    df = df.set_index("date")

    # Generate signals
    # BUY base (sell quote): base undervalued AND quote overvalued
    # SELL base (buy quote): spread normalises (crosses zero)
    df["signal"] = 0
    df.loc[
        (df["base_momentum_ratio"] > 1 + threshold) &
        (df["quote_momentum_ratio"] < 1 - threshold),
        "signal"
    ] = 1  # long base / short quote

    df.loc[
        (df["base_momentum_ratio"] < 1 - threshold) &
        (df["quote_momentum_ratio"] > 1 + threshold),
        "signal"
    ] = -1  # short base / long quote

    # Position: hold until opposite signal (simple state machine)
    df["position"] = 0
    pos = 0
    for i in range(len(df)):
        if df["signal"].iloc[i] != 0:
            pos = df["signal"].iloc[i]
        df.iloc[i, df.columns.get_loc("position")] = pos

    # Returns: when in position, we earn the base return minus the quote return
    df["base_return"] = df["base_close"].pct_change()
    df["quote_return"] = df["quote_close"].pct_change()
    df["strategy_return"] = df["position"].shift(1) * (
        df["base_return"] - df["quote_return"]
    )
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # Cumulative returns
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cumulative_base"] = (1 + df["base_return"].fillna(0)).cumprod()
    df["cumulative_quote"] = (1 + df["quote_return"].fillna(0)).cumprod()

    # The pair ratio (base/quote) — this is what you'd actually trade
    df["pair_ratio"] = df["base_close"] / df["quote_close"]
    df["pair_ratio_norm"] = df["pair_ratio"] / df["pair_ratio"].iloc[0]

    return df


# ── Reporting ────────────────────────────────────────────────────────

def print_report(df, base_symbol, quote_symbol):
    """Print backtest results."""
    if df.empty:
        print("No results to report.")
        return

    total_days = len(df)
    days_long = (df["position"] == 1).sum()
    days_short = (df["position"] == -1).sum()
    days_flat = (df["position"] == 0).sum()

    total_return = (df["cumulative_strategy"].iloc[-1] - 1) * 100
    base_return = (df["cumulative_base"].iloc[-1] - 1) * 100
    quote_return = (df["cumulative_quote"].iloc[-1] - 1) * 100

    # Count trades (position changes)
    trades = (df["position"].diff().abs() > 0).sum()

    # Sharpe ratio (annualised, assuming 365 trading days for crypto)
    daily_returns = df["strategy_return"]
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)
              if daily_returns.std() > 0 else 0)

    # Max drawdown
    cummax = df["cumulative_strategy"].cummax()
    drawdown = (df["cumulative_strategy"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    print("=" * 65)
    print("CRYPTO MOMENTUM PAIRS BACKTEST")
    print("=" * 65)
    print(f"  Pair:             {base_symbol} / {quote_symbol}")
    print(f"  Period:           {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days:     {total_days}")
    print(f"  Total trades:     {trades}")
    print()
    print(f"  Days long {base_symbol[:3]}:  {days_long:5d} ({days_long/total_days*100:.1f}%)")
    print(f"  Days long {quote_symbol[:3]}:  {days_short:5d} ({days_short/total_days*100:.1f}%)")
    print(f"  Days flat:        {days_flat:5d} ({days_flat/total_days*100:.1f}%)")
    print()
    print(f"  Strategy return:  {total_return:+.1f}%")
    print(f"  {base_symbol} buy & hold:   {base_return:+.1f}%")
    print(f"  {quote_symbol} buy & hold:  {quote_return:+.1f}%")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Max drawdown:     {max_dd:.1f}%")
    print()


def plot_results(df, base_symbol, quote_symbol, tag=""):
    """Generate charts for the backtest."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Momentum ratios
    ax = axes[0]
    ax.plot(df.index, df["base_momentum_ratio"], label=f"{base_symbol} momentum ratio",
            color="blue", alpha=0.7)
    ax.plot(df.index, df["quote_momentum_ratio"], label=f"{quote_symbol} momentum ratio",
            color="orange", alpha=0.7)
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(df.index, df["base_momentum_ratio"], df["quote_momentum_ratio"],
                    where=df["base_momentum_ratio"] > df["quote_momentum_ratio"],
                    alpha=0.15, color="green", label=f"Favour {base_symbol}")
    ax.fill_between(df.index, df["base_momentum_ratio"], df["quote_momentum_ratio"],
                    where=df["base_momentum_ratio"] < df["quote_momentum_ratio"],
                    alpha=0.15, color="red", label=f"Favour {quote_symbol}")
    ax.set_ylabel("Momentum Ratio")
    ax.set_title("Momentum Ratios (>1 = undervalued, <1 = overvalued)")
    ax.legend(loc="upper left", fontsize=8)

    # Plot 2: Position and pair ratio
    ax = axes[1]
    ax.plot(df.index, df["pair_ratio_norm"], label=f"{base_symbol}/{quote_symbol} ratio",
            color="purple", alpha=0.7)
    # Shade position
    ax2 = ax.twinx()
    ax2.fill_between(df.index, df["position"], 0, alpha=0.1, color="green",
                     where=df["position"] > 0, label="Long " + base_symbol[:3])
    ax2.fill_between(df.index, df["position"], 0, alpha=0.1, color="red",
                     where=df["position"] < 0, label="Long " + quote_symbol[:3])
    ax2.set_ylabel("Position")
    ax2.set_ylim(-2, 2)
    ax.set_ylabel("Pair Ratio (normalised)")
    ax.set_title("Pair Ratio & Position")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # Plot 3: Cumulative returns
    ax = axes[2]
    ax.plot(df.index, df["cumulative_strategy"], label="Strategy",
            color="green", linewidth=2)
    ax.plot(df.index, df["cumulative_base"], label=f"{base_symbol} hold",
            color="blue", alpha=0.5)
    ax.plot(df.index, df["cumulative_quote"], label=f"{quote_symbol} hold",
            color="orange", alpha=0.5)
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative Return (1.0 = break even)")
    ax.set_title("Cumulative Returns")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Crypto Momentum Pairs: {base_symbol} / {quote_symbol}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out = os.path.join(SCRIPT_DIR, f"backtest_crypto{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crypto momentum pairs trading backtest"
    )
    parser.add_argument(
        "--base", default="ETH-USD",
        help="Base asset (we buy this when undervalued). Default: ETH-USD"
    )
    parser.add_argument(
        "--quote", default="BTC-USD",
        help="Quote asset (we sell this when overvalued). Default: BTC-USD"
    )
    parser.add_argument(
        "--lookback", type=int, default=180,
        help="Momentum lookback window in days. Default: 180"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.03,
        help="Momentum ratio threshold to trigger a trade. Default: 0.03 (3%%)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Simulation start date (YYYY-MM-DD). Default: 2 years ago"
    )
    parser.add_argument(
        "--end", default=None,
        help="Simulation end date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate charts"
    )

    args = parser.parse_args()

    end_date = (datetime.strptime(args.end, "%Y-%m-%d")
                if args.end else datetime.now())
    start_date = (datetime.strptime(args.start, "%Y-%m-%d")
                  if args.start else end_date - timedelta(days=730))

    print()
    print(f"  Base asset:    {args.base}")
    print(f"  Quote asset:   {args.quote}")
    print(f"  Lookback:      {args.lookback} days")
    print(f"  Threshold:     {args.threshold:.1%}")
    print(f"  Period:        {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print()

    df = simulate_pairs_trade(
        base_symbol=args.base,
        quote_symbol=args.quote,
        lookback_days=args.lookback,
        threshold=args.threshold,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return

    print_report(df, args.base, args.quote)

    if args.plot:
        tag = f"_{args.base.replace('-', '')}_{args.quote.replace('-', '')}"
        plot_results(df, args.base, args.quote, tag=tag)

    # Save results
    csv_out = os.path.join(SCRIPT_DIR, "backtest_crypto_results.csv")
    df.to_csv(csv_out)
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
