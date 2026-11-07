-- | Tune the two-sided stencil on PC1 (the market factor), honestly.
--
-- One series, so the split is purely temporal (the PCA showed a by-coin
-- split is near-cosmetic -- everything shares PC1). For each stencil
-- config (field x contrast) the half-window k is chosen by walk-forward
-- Sharpe on the validation year, then the config is judged ONCE on the
-- holdout. We report the honest 'walk' result, the look-ahead 'edge'
-- ceiling, and buy & hold, so the forecasting cost is visible.
module Main (main) where

import Finance
import Boundary (causalTrend)
import Stencil
import Control.Monad (forM, forM_)
import Text.Printf (printf)

lookbackDays, driftWin :: Int
lookbackDays = 180
driftWin     = 20

kCandidates :: [Int]
kCandidates = [10, 20, 40, 60]

betaCandidates :: [Double]
betaCandidates = [0.0, 0.02, 0.05, 0.10]

holdoutStart, valStart :: String
holdoutStart = "2025-05-01"
valStart     = "2024-05-01"

sliceCell :: String -> String -> [Bar] -> [Bool] -> ([Bar], [Bool])
sliceCell lo hi bars sigs =
  unzip [ (b, s) | (b, s) <- zip bars sigs
                 , barDate b >= lo, null hi || barDate b < hi ]

-- backtest a signal generator on a date window
evalOn :: [Bar] -> String -> String -> ([Bar] -> [Bool]) -> Report
evalOn bars lo hi sig = let (bs, ss) = sliceCell lo hi bars (sig bars) in backtest bs ss

main :: IO ()
main = do
  csv <- readFile "data/PC1.csv"
  let bars = parseCSV csv
  printf "== two-sided stencil on PC1 :: %d days %s -> %s ==\n"
    (length bars) (barDate (head bars)) (barDate (last bars))
  printf "lookback %d, driftWin %d, holdout from %s\n\n" lookbackDays driftWin holdoutStart

  let bh = buyHold (fst (sliceCell holdoutStart "" bars (replicate (length bars) True)))
  printf "buy & hold (PC1, holdout):           Sharpe %5.2f  ret %+7.1f%%  maxDD %6.1f%%\n"
    (repSharpe bh) (repTotalRet bh * 100) (repMaxDD bh * 100)
  let causalRef = evalOn bars holdoutStart "" (extendT (causalTrend 20))
  printf "causal price-trend k=20 (holdout):   Sharpe %5.2f  ret %+7.1f%%  maxDD %6.1f%%\n\n"
    (repSharpe causalRef) (repTotalRet causalRef * 100) (repMaxDD causalRef * 100)

  printf "%-10s | %-24s | %-30s | %s\n"
    "config" "beta=0 baseline (holdout)" "beta* reversion (holdout)" "edge"
  putStrLn (replicate 86 '-')
  forM_ allConfigs $ \cfg -> do
    -- validation Sharpe over the (k, beta) grid
    grid <- forM [ (k, b) | k <- kCandidates, b <- betaCandidates ] $ \(k, b) -> do
      let r = evalOn bars valStart holdoutStart (walkStencil cfg driftWin lookbackDays b k)
      return (repSharpe r, (k, b))
    let -- best overall (k, beta)
        (_, (kS, bS)) = maximum grid
        -- best with reversion switched off, for a like-for-like baseline
        (_, (k0, _))  = maximum [ g | g@(_, (_, b)) <- grid, b == 0.0 ]
        walk0 = evalOn bars holdoutStart "" (walkStencil cfg driftWin lookbackDays 0.0 k0)
        walkS = evalOn bars holdoutStart "" (walkStencil cfg driftWin lookbackDays bS kS)
        edgeS = evalOn bars holdoutStart "" (edgeStencil cfg driftWin lookbackDays bS kS)
    printf "%-10s | k=%2d S %5.2f ret %+6.1f%% | k=%2d b=%.2f S %5.2f ret %+6.1f%% | %5.2f\n"
      (configName cfg)
      k0 (repSharpe walk0) (repTotalRet walk0 * 100)
      kS bS (repSharpe walkS) (repTotalRet walkS * 100)
      (repSharpe edgeS)
  putStrLn ""
  putStrLn "beta=0 = drift-only (causal in disguise); beta>0 injects the volume-built pool price m."
  putStrLn "If beta* columns beat beta=0 out-of-sample, the two-sided-ness is finally doing work."
