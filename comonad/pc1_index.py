"""
Build PC1 (the market factor) as a tradeable price+volume series.

The PCA showed PC1 is 62.5% of all variance with near-equal loadings --
the market factor. We turn it into one synthetic asset the Haskell tape
harness can trade:

  - keep coins still trading today (drop delisted/migrated ones, e.g. FTM,
    MATIC, so the common window reaches the present and the 2025-26
    temporal hold-out is usable),
  - estimate PC1 loadings on the survivors' standardised returns,
  - index return_t = Σ w_i r_i,t   with w_i ∝ PC1 loading (all positive),
    on *raw* returns -> a real, positive-weight market portfolio,
  - price = 100·exp(cumsum(index return)); volume = total $-volume.

Writes pc1_index.csv in the Bar CSV format the Haskell side reads.

Note: PC1 loadings are estimated on the full sample (a small, stable
look-ahead -- the market factor's shape barely changes with window). A
rolling-PCA refinement would remove even that; deferred as negligible.
"""

import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STILL_TRADING_AFTER = "2026-10-01"


def load():
    series = {}
    for path in sorted(glob.glob(os.path.join(HERE, "data", "*.csv"))):
        sym = os.path.basename(path)[:-4]
        rows = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[r["date"]] = (float(r["close"]), float(r["volume"]))
        series[sym] = rows
    return series


def main():
    series = load()
    survivors = {s: d for s, d in series.items() if max(d) >= STILL_TRADING_AFTER}
    dropped = sorted(set(series) - set(survivors))
    syms = sorted(survivors)
    print(f"{len(series)} coins; dropped {len(dropped)} not trading after "
          f"{STILL_TRADING_AFTER}: {dropped}")

    common = set.intersection(*(set(d) for d in survivors.values()))
    dates = sorted(common)
    print(f"{len(syms)} survivors, {len(dates)} common days: {dates[0]} -> {dates[-1]}\n")

    closes = np.array([[survivors[s][dt][0] for s in syms] for dt in dates])    # [T,N]
    vols   = np.array([[survivors[s][dt][1] for s in syms] for dt in dates])
    rets = np.diff(np.log(closes), axis=0)                                      # [T-1,N]

    z = (rets - rets.mean(0)) / rets.std(0, ddof=1)
    corr = np.cov(z, rowvar=False)
    w, V = np.linalg.eigh(corr)
    i1 = int(np.argmax(w))
    v1 = V[:, i1]
    if v1.sum() < 0:
        v1 = -v1
    evr1 = w[i1] / w.sum()
    weights = v1 / v1.sum()                       # positive, sum to 1
    print(f"PC1 explains {evr1*100:.1f}% of variance on survivors; "
          f"weight range [{weights.min():.3f}, {weights.max():.3f}]")

    idx_ret = rets @ weights                      # raw-return market portfolio
    price = 100.0 * np.exp(np.cumsum(idx_ret))
    dollar_vol = (vols * closes).sum(axis=1)[1:]  # total $-volume, aligned to rets

    out = os.path.join(HERE, "data", "PC1.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["date", "open", "high", "low", "close", "volume"])
        for dt, p, dv in zip(dates[1:], price, dollar_vol):
            wtr.writerow([dt, f"{p:.6f}", f"{p:.6f}", f"{p:.6f}", f"{p:.6f}", f"{dv:.2f}"])
    print(f"wrote {len(price)} days to {out}  (start {price[0]:.1f} -> end {price[-1]:.1f})")


if __name__ == "__main__":
    main()
