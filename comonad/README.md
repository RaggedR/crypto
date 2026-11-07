# Finance as a tape comonad

A backtester built on the **tape comonad** from `../../harness-worker-orchestration/aladdin/tape.pdf`.
Finance is a time series; a time series is a stream; a stream is a comonad.
An investment strategy is a **local evolution rule** — a co-Kleisli arrow
`w Bar -> Decision` — and `extend` runs it at every day of history at once.
That *is* the backtest.

> **Full chronological results, including the failures, are in
> [`EXPERIMENTS.md`](EXPERIMENTS.md)** — read that for the real state. This
> README explains the design; the notebook records what we found.
>
> **Status: the headline result was REFUTED by a clean out-of-sample test.**
> On PC1 (the series we tuned on), `p/slope` + reversion-to-`m` returned +77.9%
> vs a −38% market — but the design was chosen while watching PC1, so we froze
> it (commit 7b579f2) and ran it once on **unseen** assets (8 coins never in the
> universe, + a market factor built from them). It did **not** reproduce:
> reversion averaged Sharpe −0.09 on unseen coins (worst of three) and −51.5%
> return on the fresh market factor. The β=0 baseline — which *lost* −52% on PC1
> — was as good or better everywhere out-of-sample. **The PC1 +78% is an
> overfitting artifact, not a validated edge.** See [`EXPERIMENTS.md`](EXPERIMENTS.md) §8–9.
>
> **What held up:** the comonad framing and harness are sound; the discipline
> (commit the record, then run the falsifying test) worked — it caught the
> overfit. The only durable property of the bookmaker model across the project
> is drawdown/exit timing, not return generation.
>
> **In dollars:** $10,000 into the strategy out-of-sample would have ended at
> **~$5,000** (lost ~half) — *less* than buy-and-hold (~$1,700–$2,700), but still
> a loss. The outperformance is pure crash-avoidance (in cash ~half the time),
> which the β=0 rule also gives; it is not the refuted reversion edge. Damage
> control, not a thesis. See [`EXPERIMENTS.md`](EXPERIMENTS.md) §9.

## The one idea, three readings

The only thing that changes between a live strategy, a research upper
bound, and a forecast is **which way the local rule looks**:

| Reading | Comonad | Rule type | Tradeable? | What it is |
|---|---|---|---|---|
| **Causal** | `Stream` (past only) | `Stream Bar -> Bool` | ✅ yes | Robin's bookmaker-pool momentum model (`../concepts.pdf` §10). Cannot read the future — structurally. |
| **Forecast (A)** | `Tape`, right side *predicted* | `Stream Bar -> Bool` | ✅ yes | Evolve a model forward, then act on the forecast. The honest reading of "reads past and future". |
| **Oracle (B)** | `Tape`, right side *real* | `Tape Bar -> Bool` | ❌ no | Perfect-foresight ceiling: hold iff tomorrow closes higher. Measures what's on the table, never tradeable. |

`Stream` and `Tape` are the two handednesses of one comonad. The causal
rule is a `Stream` arrow; the oracle is the same shape pointed the other
way (`Tape`, reading the right neighbour). The Python `.shift(1)`
execution lag is itself just *one more step back* — another co-Kleisli move.

## Files

- `src/Tape.hs` — the `Tape` and `Stream` comonads, transcribed verbatim
  from `tape.tex` (`extract`, `duplicate`, `extend`, `=<=`, `cokPow`).
- `test/Golden.hs` — correctness anchor: reproduces the exact impulse
  (heat) and bump-splitting (wave) numeric traces printed in `tape.tex`.
- `src/Finance.hs` — `Bar`/CSV, the bookmaker rule, the three strategies,
  and `backtest` (equity curve, CAGR, Sharpe, max drawdown, time-in-market).
- `src/Forward.hs` — the return-regression forward model (OLS). *Superseded*
  as a strategy (it failed out-of-sample; see notebook §2), kept for the record.
- `src/Boundary.hs` — the forward map as a **boundary extrapolator**
  (`forecastFrom` = drift; `forecastReversion` = drift + reversion to `m`) and
  the centred/causal trend rules.
- `src/Stencil.hs` — the configurable two-sided stencil (field ∈ {p,m,g} ×
  contrast ∈ {slope,curv} × window k), with honest `walk` and look-ahead `edge`.
- `app/Main.hs` (`comonad`) — three-readings table on one asset.
- `app/Fit.hs` (`fit`) — return-regression study with train/validate/test.
- `app/Edge.hs` (`edge`) — centred zero-lag rule across the coin universe.
- `app/PC1.hs` (`pc1`) — stencil tuning on PC1 with the β (reversion) sweep.
- `app/WalkForward.hs` (`walkforward`) — multi-regime walk-forward + bottom line.
- `pca.py`, `pc1_index.py` — PCA of the universe and the PC1 index builder.
- `fetch.py`, `fetch_universe.sh` — CCXT → CSV bridge (Binance; real volume).
- **[`EXPERIMENTS.md`](EXPERIMENTS.md)** — the lab notebook: every run, incl. failures.

## Run

```bash
# data (once)
../venv/bin/python fetch.py --symbol BTC/USDT --since 2017-08-01   # one coin
./fetch_universe.sh                                                # the 30-coin universe
../venv/bin/python pca.py            # PCA: PC1 = 62.5% var, eff. factors ≈ 2.5
../venv/bin/python pc1_index.py      # build data/PC1.csv (market factor)

cabal test                           # comonad golden traces + OLS recovery
cabal run comonad -- data/BTCUSDT.csv 180 20   # three readings on BTC
cabal run fit                        # return-regression study (FAILED, §2)
cabal run pc1                        # two-sided stencil on PC1, β sweep (§5–6)
cabal run walkforward                # multi-regime walk-forward + bottom line (§7–8)
```

## Single-asset illustration (BTC/USDT, 2017-08-17 → 2026-11-07, Binance)

*(Context, not the headline — the validated result is the PC1 walk-forward above.)*

```
buy & hold        ret   +1670%  Sharpe 0.80  maxDD -83%  in-mkt 100%
causal (live)     ret    +379%  Sharpe 0.71  maxDD -54%  in-mkt  25%   <- cuts drawdown by 30pts
forecast (A)      ret   +2521%  Sharpe 0.92  maxDD -68%  in-mkt  66%
oracle (B)        ret  ~3e19 %   Sharpe 10.5  maxDD   0%  in-mkt  51%   <- the ceiling, not an investment
```

The causal rule reproduces the repo's headline behaviour — **drawdown
reduction is the real edge** (−83% → −54%), at the cost of return, by
sitting in cash 75% of the time. The oracle shows how enormous the gap to
perfect foresight is: that gap is the entire research problem.

## The forward model (Reading A), fit without fishing

`src/Forward.hs` + `app/fit` learn the "evolve the system forward" step as
a linear equation of motion for log price, with the momentum field as
forcing:

```
Δp_{t+h}  =  c  +  α·v  +  β·g  +  γ·a
  v = p_t − p_{t−1}              price velocity   (inertia / momentum)
  g = m_t − p_t                  gap to the pool  (reversion to momentum price)
  a = m_t − 2m_{t−1} + m_{t−2}   momentum accel   ([1,−2,1], the tape.tex Laplacian)
```

**Anti-fishing protocol** (both axes held out, OLS not search):

- Coins split TRAIN (22) / TEST (8) by a deterministic hash of the symbol.
- Time split at `2025-05-01`; the recent ~18 months is a final hold-out.
- θ fit by **OLS** (convex, closed-form, nothing to search) on
  TRAIN×subfit; the only hyperparameter, horizon `h`, chosen by validation
  Sharpe on TRAIN×[2024-05,2025-05); final θ refit on all TRAIN×in-time.
- The fitted law is evaluated **once** on TRAIN×holdout, TEST×in-time,
  TEST×holdout — none ever seen during fitting.

**Result — the forward model does NOT beat the causal rule.** Run `cabal
run fit`. The fitted law is

```
Δp_{t+40} = +0.0067 + 0.227·v − 0.011·g − 0.024·a
```

- Only the **inertia** term `v` is real; acting on it is just
  trend-following.
- The **gap** term `g` — the heart of the bookmaker thesis — has a
  ~zero/negative coefficient and **negative out-of-sample R²**. The pool
  gap does not *linearly* predict future returns.
- Out of sample, forward (A) underperforms the causal rule everywhere and
  gives back the drawdown (it's in-market most of the time).

This is the protocol working as designed: a Sharpe grid-search would have
"found" the +0.53 validation Sharpe and declared victory; the held-out
cells kill it. **The causal rule's edge was never return-forecasting — it
is drawdown reduction via regime filtering** (stay in cash when money is
leaving), which a return-regression cannot capture.

## Honest caveats

- These numbers are on a **different window and data source** (Binance
  2017–2026) than `../concepts.pdf` (yfinance, April 2021 start), so they
  are *not* a line-for-line reproduction of those figures — they are the
  same model on cleaner, longer, honest-volume data.
- **Forecast (A)** here uses a trivial drift forecast, so it reduces to
  simple trend-following; beating buy-and-hold on a strongly trending asset
  is not yet a result. The real work is a better forward model (the "evolve
  the system forward" step). This is the open frontier, not a finding.
- The oracle's astronomical return is correct and meaningless by design.
