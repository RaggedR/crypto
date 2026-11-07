"""
PCA on the crypto universe's daily log-returns.

Why: the two-sided rule will be tuned on these coins, so we need to know
how many *independent* directions the universe actually has. Highly
correlated coins mean the effective sample size for honest validation is
far below the coin count -- the real anti-fishing constraint.

Aligns all data/*.csv on their common date range, standardises returns,
and eigendecomposes the correlation matrix. Prints the variance spectrum,
the effective number of factors (participation ratio), and the loadings of
the top components. Saves the top-k PC score series to pc_scores.csv.
"""

import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 5


def load():
    series = {}
    for path in sorted(glob.glob(os.path.join(HERE, "data", "*.csv"))):
        sym = os.path.basename(path)[:-4]
        d = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                d[row["date"]] = float(row["close"])
        series[sym] = d
    return series


def main():
    series = load()
    syms = sorted(series)
    # common date range across ALL coins
    common = set.intersection(*(set(d) for d in series.values()))
    dates = sorted(common)
    print(f"{len(syms)} coins, {len(dates)} common days: {dates[0]} -> {dates[-1]}\n")

    closes = np.array([[series[s][dt] for s in syms] for dt in dates])   # [T, N]
    rets = np.diff(np.log(closes), axis=0)                               # [T-1, N]

    # standardise each coin's return column -> correlation-matrix PCA
    mu, sd = rets.mean(0), rets.std(0, ddof=1)
    z = (rets - mu) / sd
    corr = np.cov(z, rowvar=False)                                       # [N, N]

    w, V = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    evr = w / w.sum()
    part_ratio = (w.sum() ** 2) / (w ** 2).sum()                         # effective # factors

    print("variance spectrum (top 10):")
    print("  PC   eigenvalue   explained   cumulative")
    cum = 0.0
    for i in range(min(10, len(w))):
        cum += evr[i]
        print(f"  {i+1:2d}   {w[i]:9.3f}   {evr[i]*100:7.1f}%   {cum*100:7.1f}%")
    print(f"\neffective # of independent factors (participation ratio): {part_ratio:.2f}")
    print(f"  (vs {len(syms)} coins -- that gap is the honest validation budget)\n")

    for pc in range(min(3, len(w))):
        load_pc = V[:, pc]
        if load_pc.sum() < 0:          # sign convention: PC1 is the market, loadings positive
            load_pc = -load_pc
        pairs = sorted(zip(syms, load_pc), key=lambda t: -abs(t[1]))
        tag = {0: "PC1 (market factor)", 1: "PC2", 2: "PC3"}[pc]
        print(f"{tag}  [{evr[pc]*100:.1f}% var]  top loadings:")
        for s, l in pairs[:8]:
            print(f"    {s:10s} {l:+.3f}")
        print()

    # save top-k PC score time series for downstream use as features
    scores = z @ V[:, :TOPK]                                            # [T-1, TOPK]
    out = os.path.join(HERE, "pc_scores.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["date"] + [f"PC{i+1}" for i in range(TOPK)])
        for dt, rowv in zip(dates[1:], scores):
            wtr.writerow([dt] + [f"{x:.6f}" for x in rowv])
    print(f"saved top-{TOPK} PC score series to {out}")


if __name__ == "__main__":
    main()
