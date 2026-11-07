-- | The forward map as a /boundary extrapolator/, and the strategy as a
-- /two-sided/ (zero-lag) tape rule.
--
-- A centred rule reads k steps left and k steps right of the focus. In
-- the interior of history both sides are real. At the live edge -- today
-- -- the right side does not exist yet, so the rule cannot fire exactly
-- where we need a decision. The *forward map* fixes this: evolve the
-- series a few steps forward ("in the usual way") to synthesise the
-- missing right side, then the two-sided rule reads past + forecast and
-- we @extract@ a decision at today.
--
-- Two evaluation modes, and the gap between them is the whole point:
--
--   * 'edgeDecisions'  — real future in the interior, forecast only for
--     the final k days. The *ideal* the rule would achieve if forecasts
--     were perfect. (Look-ahead in the interior; not live-reproducible.)
--   * 'walkDecisions'  — forecast the right side at *every* step from data
--     up to that day only. Causal, live-reproducible. What you actually get.
module Boundary
  ( forecastFrom
  , forecastReversion
  , centeredTrend
  , causalTrend
  , edgeDecisions
  , walkDecisions
  , liveDecision
  ) where

import Tape
import Finance (Bar(..), momentumPrice)
import Finance (extendT)

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

logClose :: Bar -> Double
logClose = log . barClose

-- | Forward map: synthesise the next @k@ bars from a nearest-first history
-- @[focus, prev, ...]@. "Momentum continues": extrapolate log-price at the
-- recent mean log-return (drift over @driftWin@ days). Volume held at the
-- recent mean so the bars are well-formed. This is the extrapolator, not
-- the strategy -- deliberately simple.
forecastFrom :: Int -> Int -> [Bar] -> [Bar]
forecastFrom driftWin k histNF =
  [ synth (p0 + fromIntegral j * mu) | j <- [1 .. k] ]
  where
    closes = map barClose histNF                       -- nearest first
    lrets  = zipWith (\new old -> log new - log old)    -- recent log returns
               closes (drop 1 closes)
    mu     = mean (take driftWin lrets)
    p0     = log (head closes)
    vol    = mean (take driftWin (map barVolume histNF))
    synth p = let c = exp p in Bar "" c c c c vol

-- | Richer forward map: drift PLUS reversion of log-price toward the
-- current pool price @m_t@. One step is
--
--   p_{j} = p_{j-1} + μ + β·(log m_t − p_{j-1})
--
-- so the forecast relaxes geometrically toward @log m_t@ at rate β. The
-- crucial point: @m_t@ is built from *volume* (order flow), so it is NOT a
-- function of the trailing price window -- with β>0 the forecast future
-- side finally carries information the past price does not, and the
-- two-sided rule stops collapsing to a causal one. β=0 recovers
-- 'forecastFrom' exactly.
forecastReversion :: Int -> Int -> Int -> Double -> [Bar] -> [Bar]
forecastReversion driftWin lookback k beta histNF =
  map synth (take k (drop 1 (iterate step p0)))
  where
    closes = map barClose histNF
    lrets  = zipWith (\new old -> log new - log old) closes (drop 1 closes)
    mu     = mean (take driftWin lrets)
    p0     = log (head closes)
    vol    = mean (take driftWin (map barVolume histNF))
    logm   = case momentumPrice (reverse (take (lookback + 1) histNF)) of
               Just mp | mp > 0 -> log mp
               _                -> p0          -- undefined early: fall back to drift only
    step p = p + mu + beta * (logm - p)
    synth p = let c = exp p in Bar "" c c c c vol

-- | Zero-lag strategy: hold iff the mean log-price over the k steps to the
-- /right/ (future) exceeds the mean over the k steps to the /left/ (past)
-- -- a centred trend through the focus, with no moving-average lag.
centeredTrend :: Int -> Tape Bar -> Bool
centeredTrend k t = mean rs > mean ls
  where ls = map logClose (drop 1 (takeL (k + 1) t))   -- k past (focus excluded)
        rs = map logClose (drop 1 (takeR (k + 1) t))   -- k future (focus excluded)

-- | Causal counterpart (the lagging baseline): hold iff today's log-price
-- is above the trailing k-day mean. Reads only the left side.
causalTrend :: Int -> Tape Bar -> Bool
causalTrend k t = logClose (focus t) > mean (map logClose (drop 1 (takeL (k + 1) t)))
  where focus (Tape _ x _) = x

--------------------------------------------------------------------------
-- Running the two-sided rule over history, two ways.
--------------------------------------------------------------------------

-- | IDEAL: append k forecast bars once at the end, then run the centred
-- rule at every real day. Interior days read the real future (look-ahead);
-- only the last k read the forecast. One decision per real bar.
edgeDecisions :: Int -> Int -> [Bar] -> [Bool]
edgeDecisions driftWin k bars =
  take (length bars) (extendT (centeredTrend k) extended)
  where extended = bars ++ forecastFrom driftWin k (reverse bars)

-- | HONEST: at every day forecast the right side from data up to that day
-- only, then apply the centred rule. Causal and live-reproducible.
-- O(n·driftWin) via incremental history. One decision per bar.
walkDecisions :: Int -> Int -> [Bar] -> [Bool]
walkDecisions driftWin k bars = go [] bars
  where
    go _  []       = []
    go ls (x : rs) = centeredTrend k (Tape ls x synth) : go (x : ls) rs
      where synth = forecastFrom driftWin k (x : ls)   -- right side is forecast

-- | The live decision for today: the centred rule at the final bar, whose
-- right side is pure forecast. This is the payoff of the construction --
-- a zero-lag rule evaluated at the very edge.
liveDecision :: Int -> Int -> [Bar] -> Bool
liveDecision driftWin k bars = case reverse bars of
  []            -> False
  hist@(x : ls) -> centeredTrend k (Tape ls x (forecastFrom driftWin k hist))
