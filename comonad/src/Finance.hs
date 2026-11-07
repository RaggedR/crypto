-- | Finance as a comonad: OHLCV bars on a tape, strategies as co-Kleisli
-- arrows, the backtest as @extend@.
--
-- Three readings of the same machinery, distinguished *only* by which way
-- the local rule looks:
--
--   * 'causalSignal'  — a 'Stream' rule (past only). Live-tradeable. This
--     is Robin's bookmaker-pool momentum model from @concepts.pdf@ §10.
--   * 'oracleSignal'  — a 'Tape' rule that reads the *actual* future.
--     NOT tradeable; it is the perfect-foresight upper bound (Reading B).
--   * 'forecastSignal' — a 'Tape' rule whose right side is a *forecast*
--     built from the past (Reading A). Tradeable; see 'forecastBars'.
--
-- All three produce one decision per day; 'backtest' folds decisions and
-- next-day returns into an equity curve with the same one-day execution
-- lag the Python code uses (@position = signal.shift(1)@).
module Finance
  ( Bar(..)
  , parseCSV
    -- * the momentum field
  , momentumPrice
  , momentumRatio
    -- * strategies (co-Kleisli arrows)
  , causalSignal
  , oracleSignal
  , forecastSignal
    -- * running a rule over all of history (finite @extend@)
  , extendS
  , extendT
    -- * backtest
  , Report(..)
  , backtest
  , buyHold
  , formatReport
  ) where

import Tape
import Data.List (foldl')
import Text.Printf (printf)

--------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------

data Bar = Bar
  { barDate   :: !String
  , barOpen   :: !Double
  , barHigh   :: !Double
  , barLow    :: !Double
  , barClose  :: !Double
  , barVolume :: !Double
  } deriving (Show)

-- | Parse a @date,open,high,low,close,volume@ CSV (header row skipped).
parseCSV :: String -> [Bar]
parseCSV = map row . drop 1 . filter (not . null) . lines
  where
    row l = case splitOn ',' l of
      (d:o:h:lo:c:v:_) -> Bar d (rd o) (rd h) (rd lo) (rd c) (rd v)
      _                -> error ("bad CSV row: " ++ l)
    rd s = read s :: Double

splitOn :: Char -> String -> [String]
splitOn sep = foldr step [""]
  where step c acc@(cur:rest) | c == sep  = "" : acc
                              | otherwise = (c:cur) : rest
        step _ [] = [""]

--------------------------------------------------------------------------
-- Finite @extend@: run a co-Kleisli arrow at every day of a finite history.
--------------------------------------------------------------------------

-- | The finite realization of @extend@ for a *causal* 'Stream': at each
-- day, focus the stream there (with all earlier days as its past) and run
-- the rule. Returns one result per day, chronological. O(n) via sharing.
extendS :: (Stream a -> b) -> [a] -> [b]
extendS _ [] = []
extendS f (x0:xs) = map f (scanl step (Stream [] x0) xs)
  where step (Stream ps x) x' = Stream (x:ps) x'

-- | The finite realization of @extend@ for a two-sided 'Tape': at each
-- day the focus sees the real past on its left and the real future on its
-- right. Returns one result per day, chronological. O(n) via sharing.
extendT :: (Tape a -> b) -> [a] -> [b]
extendT f = go []
  where go _  []       = []
        go ls (x:rs)   = f (Tape ls x rs) : go (x:ls) rs

--------------------------------------------------------------------------
-- The bookmaker-pool momentum model (concepts.pdf §10)
--------------------------------------------------------------------------

-- | Pool-implied (\"momentum\") price over a chronological window (oldest
-- first, focus last). Weights each day's return by volume/mean-volume and
-- runs the bookmaker pool. Faithful to
-- @momentum_single.compute_rolling_momentum@.
momentumPrice :: [Bar] -> Maybe Double
momentumPrice ws
  | length ws < 11 = Nothing            -- Python needs >= 10 returns
  | meanVol == 0   = Nothing
  | otherwise      = Just mp
  where
    closes   = map barClose ws
    rets     = zipWith (\a b -> (b - a) / a) closes (tail closes)
    vols     = map barVolume (tail ws)
    meanVol  = sum vols / fromIntegral (length vols)
    netFlows = zipWith (\v r -> (v / meanVol) * r) vols rets
    pool0    = head closes
    mp       = foldl' (\m nf -> m + nf * m) pool0 netFlows

-- | Momentum ratio: pool-implied price over actual price. > 1 means money
-- is still flowing in (hold); <= 1 means it is leaving (cash).
momentumRatio :: [Bar] -> Maybe Double
momentumRatio ws = fmap (\mp -> mp / barClose (last ws)) (momentumPrice ws)

-- | Live, causal strategy: hold iff the momentum ratio over the last
-- @lookback@ days exceeds 1. A 'Stream' co-Kleisli arrow — structurally
-- unable to read the future.
causalSignal :: Int -> Stream Bar -> Bool
causalSignal lookback s =
  case momentumRatio (reverse (takeS (lookback + 1) s)) of
    Just r  -> r > 1.0
    Nothing -> False

-- | Perfect-foresight oracle (Reading B): hold iff tomorrow closes higher.
-- A 'Tape' arrow reading the real right neighbour. The upper bound only.
oracleSignal :: Tape Bar -> Bool
oracleSignal (Tape _ x (r:_)) = barClose r > barClose x
oracleSignal _                = False     -- last day: no future

-- | Forecast strategy (Reading A): build a forecast of the next @horizon@
-- days from the past (geometric drift = mean of the last @lookback@ daily
-- log-ish returns), then hold iff the forecast price @horizon@ days out is
-- above today. Tradeable: the right side it reads is *predicted*, not real.
forecastSignal :: Int -> Int -> Stream Bar -> Bool
forecastSignal lookback horizon s =
  let ws = reverse (takeS (lookback + 1) s)
  in case ws of
       [] -> False
       _  | length ws < 11 -> False
          | otherwise ->
              let closes = map barClose ws
                  rets   = zipWith (\a b -> (b - a) / a) closes (tail closes)
                  drift  = sum rets / fromIntegral (length rets)
                  today  = last closes
                  future = today * (1 + drift) ^^ horizon
              in future > today

--------------------------------------------------------------------------
-- Backtest: fold decisions + next-day returns into an equity curve.
--------------------------------------------------------------------------

data Report = Report
  { repDays       :: !Int
  , repTimeIn     :: !Double   -- fraction of days holding
  , repTrades     :: !Int
  , repTotalRet   :: !Double   -- fraction, e.g. 4.63 = +463%
  , repCAGR       :: !Double
  , repSharpe     :: !Double
  , repMaxDD      :: !Double   -- negative fraction
  , repFinalMult  :: !Double   -- equity multiple on $1
  }

-- | Fold a per-day decision series against the bars, with the one-day
-- execution lag (@position[i] = signal[i-1]@), exactly as the Python does.
backtest :: [Bar] -> [Bool] -> Report
backtest bars signals = report bars positions
  where positions = False : init signals     -- shift(1): trade on yesterday's call

-- | Buy-and-hold benchmark: always in.
buyHold :: [Bar] -> Report
buyHold bars = report bars (replicate (length bars) True)

report :: [Bar] -> [Bool] -> Report
report bars positions = Report
  { repDays      = n
  , repTimeIn    = fromIntegral (length (filter id positions)) / fromIntegral n
  , repTrades    = length (filter id (zipWith (/=) positions (drop 1 positions)))
  , repTotalRet  = final - 1
  , repCAGR      = if years > 0 then final ** (1 / years) - 1 else 0
  , repSharpe    = if sd > 0 then mean / sd * sqrt 365 else 0
  , repMaxDD     = maxDD equity
  , repFinalMult = final
  }
  where
    n        = length bars
    closes   = map barClose bars
    assetRet = 0 : zipWith (\a b -> b / a - 1) closes (drop 1 closes)
    stratRet = zipWith (\p r -> if p then r else 0) positions assetRet
    equity   = drop 1 (scanl (\acc r -> acc * (1 + r)) 1 stratRet)
    final    = if null equity then 1 else last equity
    years    = fromIntegral n / 365
    mean     = sum stratRet / fromIntegral n
    sd       = let m = mean
                   var = sum [ (r - m) ^ (2 :: Int) | r <- stratRet ]
                             / fromIntegral (n - 1)      -- sample std (pandas ddof=1)
               in sqrt var
    maxDD xs = minimum (0 : zipWith (\e pk -> (e - pk) / pk) xs peaks)
      where peaks = scanl1 max xs

formatReport :: String -> Report -> String
formatReport label r = printf
  "%-16s  ret %+9.1f%%  CAGR %+6.1f%%  Sharpe %5.2f  maxDD %6.1f%%  in-mkt %4.0f%%  trades %4d  ($1->$%.2f)"
  label
  (repTotalRet r * 100) (repCAGR r * 100) (repSharpe r)
  (repMaxDD r * 100) (repTimeIn r * 100) (repTrades r) (repFinalMult r)
