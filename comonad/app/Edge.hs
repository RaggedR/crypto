-- | The two-sided strategy, evaluated honestly.
--
-- The strategy is a centred (zero-lag) trend rule on the tape; the forward
-- map only extends the right boundary so the rule can fire at the edge.
-- We compare, on held-out data:
--
--   * edge  (ideal)   — centred rule, real future in the interior
--   * walk  (honest)  — centred rule, future forecast at every step (live)
--   * causal (lagging) — one-sided trend of the same window, for contrast
--   * buy & hold
--
-- The edge/walk gap measures the cost of imperfect forecasting; the
-- walk/causal gap measures whether zero-lag actually helps once you must
-- forecast. Coins/time are split exactly as in the forward-model study.
module Main (main) where

import Finance
import Forward (isTrainCoin)
import Boundary
import Control.Monad (forM, forM_)
import Data.List (sortBy, isSuffixOf, partition)
import System.Directory (getDirectoryContents)
import Text.Printf (printf)

driftWin :: Int
driftWin = 20

kCandidates :: [Int]
kCandidates = [10, 20, 40, 60]

holdoutStart, valStart :: String
holdoutStart = "2025-05-01"
valStart     = "2024-05-01"

minCellBars :: Int
minCellBars = 60

loadCoins :: IO [(String, [Bar])]
loadCoins = do
  files <- getDirectoryContents "data"
  let csvs = filter (".csv" `isSuffixOf`) files
  forM (sortBy compare csvs) $ \f -> do
    txt <- readFile ("data/" ++ f)
    return (take (length f - 4) f, parseCSV txt)

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

sliceCell :: String -> String -> [Bar] -> [Bool] -> ([Bar], [Bool])
sliceCell lo hi bars sigs =
  unzip [ (b, s) | (b, s) <- zip bars sigs
                 , barDate b >= lo, null hi || barDate b < hi ]

reportsOnCell :: [(String, [Bar])] -> String -> String -> ([Bar] -> [Bool]) -> [Report]
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
summarise label fwd bh =
  printf "  %-20s  n=%2d  mean Sharpe %5.2f (B&H %5.2f)  mean ret %+8.1f%%  mean maxDD %6.1f%%  beat %d/%d\n"
    label (length fwd)
    (mean (map repSharpe fwd)) (mean (map repSharpe bh))
    (mean (map (\r -> repTotalRet r * 100) fwd))
    (mean (map (\r -> repMaxDD r * 100) fwd))
    (length (filter id (zipWith (\f b -> repSharpe f > repSharpe b) fwd bh)))
    (length fwd)

main :: IO ()
main = do
  coins <- loadCoins
  let (trainC, testC) = partition (isTrainCoin . fst) coins
  printf "== two-sided (zero-lag) strategy :: driftWin %d, holdout from %s ==\n\n" driftWin holdoutStart
  printf "coins: %d -> %d train, %d test\n\n" (length coins) (length trainC) (length testC)

  -- select the centred half-window k by honest (walk-forward) validation Sharpe
  putStrLn "centred-window selection (TRAIN coins, walk-forward Sharpe on validation year):"
  scored <- forM kCandidates $ \k -> do
    let reps = reportsOnCell trainC valStart holdoutStart (walkDecisions driftWin k)
        sh   = mean (map repSharpe reps)
    printf "  k=%2d : walk-forward validation Sharpe %5.2f\n" k sh
    return (sh, k)
  let bestK = snd (maximum scored)
  printf "  -> selected k = %d\n\n" bestK

  let cells =
        [ ("TRAIN x in-time", trainC, "", holdoutStart)
        , ("TRAIN x HOLDOUT", trainC, holdoutStart, "")
        , ("TEST  x in-time", testC,  "", holdoutStart)
        , ("TEST  x HOLDOUT", testC,  holdoutStart, "") ]

  putStrLn "out-of-sample evaluation (k fixed from validation):"
  forM_ cells $ \(name, cs, lo, hi) -> do
    printf "\n%s\n" name
    summarise "edge (ideal)"   (reportsOnCell cs lo hi (edgeDecisions driftWin bestK)) (bhOnCell cs lo hi)
    summarise "walk (honest)"  (reportsOnCell cs lo hi (walkDecisions driftWin bestK)) (bhOnCell cs lo hi)
    summarise "causal (lag)"   (reportsOnCell cs lo hi (extendT (causalTrend bestK)))  (bhOnCell cs lo hi)

  printf "\ntoday's live decision (centred rule at the forecast-extended edge, k=%d):\n" bestK
  forM_ coins $ \(sym, bars) ->
    printf "  %-10s %s\n" sym (if liveDecision driftWin bestK bars then "HOLD" else "cash")

-- extendT over a two-sided rule needs the Finance re-export; causalTrend
-- reads only the left, so the real right side it sees is harmless.
