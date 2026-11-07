-- | Fit the forward model honestly and report it once on held-out data.
--
-- Protocol (both coin and time hold-out, OLS fit):
--
--   * Coins split TRAIN / TEST by a deterministic hash of the symbol.
--   * Time split at @holdoutStart@: everything before is in-time; the
--     recent tail is the final temporal hold-out.
--   * θ(h) is fit by OLS on {TRAIN coins} × {subfit time} for each
--     candidate horizon h; h is chosen by mean trading Sharpe on the
--     {TRAIN coins} × {validation time} slice (out-of-sample vs subfit).
--   * The final θ is refit on all {TRAIN coins} × {in-time} rows, then
--     evaluated ONCE on the untouched cells: TRAIN×holdout (temporal),
--     TEST×in-time (cross-sectional), TEST×holdout (both).
--
-- No parameter is ever chosen by looking at a hold-out cell.
module Main (main) where

import Finance
import Forward
import Control.Monad (forM, forM_)
import Data.List (sortBy, isSuffixOf, partition)
import System.Directory (getDirectoryContents)
import Text.Printf (printf)

lookbackDays :: Int
lookbackDays = 180

horizonCandidates :: [Int]
horizonCandidates = [5, 10, 20, 40]

holdoutStart, valStart :: String
holdoutStart = "2025-05-01"   -- recent ~18 months: final temporal hold-out
valStart     = "2024-05-01"   -- the year before: horizon-selection validation

minCellBars :: Int
minCellBars = 60

--------------------------------------------------------------------------

loadCoins :: IO [(String, [Bar])]
loadCoins = do
  files <- getDirectoryContents "data"
  let csvs = filter (".csv" `isSuffixOf`) files
  forM (sortBy compare csvs) $ \f -> do
    txt <- readFile ("data/" ++ f)
    let sym = take (length f - 4) f           -- strip ".csv"
    return (sym, parseCSV txt)

-- rows whose *target* day falls in [lo, hi) (hi == "" means open end),
-- so no row ever peeks past the time boundary it is assigned to.
rowsInTime :: String -> String -> [Row] -> [Row]
rowsInTime lo hi = filter (\r -> rowTargetDate r >= lo && (null hi || rowTargetDate r < hi))

-- slice aligned (bar, signal) pairs to a date window [lo, hi)
sliceCell :: String -> String -> [Bar] -> [Bool] -> ([Bar], [Bool])
sliceCell lo hi bars sigs =
  unzip [ (b, s) | (b, s) <- zip bars sigs
                 , barDate b >= lo, null hi || barDate b < hi ]

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

--------------------------------------------------------------------------
-- Evaluate a strategy over a cell: per-coin Reports (thin cells skipped).
--------------------------------------------------------------------------

reportsOnCell
  :: [(String, [Bar])] -> String -> String -> ([Bar] -> [Bool]) -> [Report]
reportsOnCell coins lo hi signalFn =
  [ backtest bs ss
  | (_, bars) <- coins
  , let (bs, ss) = sliceCell lo hi bars (signalFn bars)
  , length bs >= minCellBars ]

bhOnCell :: [(String, [Bar])] -> String -> String -> [Report]
bhOnCell coins lo hi =
  [ buyHold bs
  | (_, bars) <- coins
  , let (bs, _) = sliceCell lo hi bars (replicate (length bars) True)
  , length bs >= minCellBars ]

summarise :: String -> [Report] -> [Report] -> IO ()
summarise label fwd bh = do
  let n       = length fwd
      beat    = length (filter id (zipWith (\f b -> repSharpe f > repSharpe b) fwd bh))
  printf "  %-26s  n=%2d  mean Sharpe %5.2f (B&H %5.2f)  mean ret %+8.1f%%  mean maxDD %6.1f%%  beat-B&H %d/%d\n"
    label n
    (mean (map repSharpe fwd)) (mean (map repSharpe bh))
    (mean (map (\r -> repTotalRet r * 100) fwd))
    (mean (map (\r -> repMaxDD r * 100) fwd))
    beat n

--------------------------------------------------------------------------

main :: IO ()
main = do
  coins <- loadCoins
  let (trainC, testC) = partition (isTrainCoin . fst) coins
  printf "== forward-model fit :: lookback %d, holdout from %s ==\n\n" lookbackDays holdoutStart
  printf "coins: %d total  ->  %d train, %d test (deterministic hash split)\n"
    (length coins) (length trainC) (length testC)
  printf "  train: %s\n" (unwords (map fst trainC))
  printf "  test : %s\n\n" (unwords (map fst testC))

  -- All feature rows per train coin, per candidate horizon (fit depends on h).
  let trainRows h = concat [ featureRows lookbackDays h bars | (_, bars) <- trainC ]

  -- Select horizon h by validation trading Sharpe (out-of-sample vs subfit).
  printf "horizon selection (fit on TRAIN coins, target < %s; validate on [%s,%s)):\n"
    valStart valStart holdoutStart
  scored <- forM horizonCandidates $ \h -> do
    let allRows = trainRows h
        subfit  = rowsInTime "" valStart allRows
        theta_h = fitOLS 0 subfit
        -- trading Sharpe on the validation period, train coins
        fwdSig  = forwardSignals theta_h lookbackDays
        valReps = reportsOnCell trainC valStart holdoutStart fwdSig
        valSh   = mean (map repSharpe valReps)
        r2val   = rSquared theta_h (rowsInTime valStart holdoutStart allRows)
    printf "  h=%2d : validation Sharpe %5.2f   R²(val) %+.4f   θ=[%s]\n"
      h valSh r2val (fmtTheta theta_h)
    return (valSh, h)
  let bestH = snd (maximum scored)
  printf "  -> selected horizon h = %d\n\n" bestH

  -- Refit the FINAL model on all in-time train rows (subfit + validation).
  let allRowsBest = trainRows bestH
      inTimeRows  = rowsInTime "" holdoutStart allRowsBest
      thetaFinal  = fitOLS 0 inTimeRows
  printf "final law (refit on all TRAIN×in-time rows, n=%d):\n" (length inTimeRows)
  printf "  Δp_{t+%d} = %+.4f + %+.4f·v %+.4f·g %+.4f·a\n"
    bestH (thetaFinal!!0) (thetaFinal!!1) (thetaFinal!!2) (thetaFinal!!3)
  printf "    v=price velocity (inertia), g=gap m-p (reversion), a=momentum accel (curvature)\n\n"

  let fwdSig  = forwardSignals thetaFinal lookbackDays
      causSig = extendS (causalSignal lookbackDays)

  -- The honest, touched-once evaluation cells.
  let cells =
        [ ("TRAIN x in-time (fit region)", trainC, "", holdoutStart)
        , ("TRAIN x HOLDOUT (temporal)",   trainC, holdoutStart, "")
        , ("TEST  x in-time (cross-sec)",  testC,  "", holdoutStart)
        , ("TEST  x HOLDOUT (both)",       testC,  holdoutStart, "") ]

  putStrLn "out-of-sample evaluation (forward Reading A vs buy & hold; causal for reference):"
  forM_ cells $ \(name, cs, lo, hi) -> do
    printf "\n%s\n" name
    summarise "forward (A)" (reportsOnCell cs lo hi fwdSig) (bhOnCell cs lo hi)
    summarise "causal (live)" (reportsOnCell cs lo hi causSig) (bhOnCell cs lo hi)
  putStrLn ""
  putStrLn "TRAIN×in-time is the fit region (expect flattery); the other three are genuinely held out."

fmtTheta :: [Double] -> String
fmtTheta = unwords . map (printf "%+.3f")
