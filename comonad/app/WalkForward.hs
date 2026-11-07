-- | Multi-regime walk-forward: does reversion-to-m (β>0) beat drift-only
-- (β=0) when the hold-out is a BULL, not just a bear?
--
-- For each 6-month hold-out window we select (k[,β]) on the preceding
-- year (validation), then judge once on the hold-out. Two arms:
--
--   * β=0  — drift-only forecast (the causal-in-disguise baseline)
--   * β>0  — forecast relaxes toward the volume-built pool price m
--
-- Each window is labelled BULL/bear by its buy-and-hold return. If β>0
-- beats β=0 only in bears, reversion is a defensiveness dial; if it also
-- helps (or at least holds) in bulls, m carries real information.
module Main (main) where

import Finance
import Stencil
import Control.Monad (forM)
import Text.Printf (printf)

lookbackDays, driftWin :: Int
lookbackDays = 180
driftWin     = 20

kCandidates :: [Int]
kCandidates = [10, 20, 40, 60]

betaPos :: [Double]
betaPos = [0.02, 0.05, 0.10]

-- (validationStart, holdoutStart, holdoutEnd); validation = the year before.
windows :: [(String, String, String)]
windows =
  [ ("2021-07-01", "2022-07-01", "2023-01-01")
  , ("2022-01-01", "2023-01-01", "2023-07-01")
  , ("2022-07-01", "2023-07-01", "2024-01-01")
  , ("2023-01-01", "2024-01-01", "2024-07-01")
  , ("2023-07-01", "2024-07-01", "2025-01-01")
  , ("2024-01-01", "2025-01-01", "2025-07-01")
  , ("2024-07-01", "2025-07-01", "2026-01-01")
  , ("2025-01-01", "2026-01-01", "2026-07-01")
  , ("2025-07-01", "2026-07-01", "")
  ]

sliceCell :: String -> String -> [Bar] -> [Bool] -> ([Bar], [Bool])
sliceCell lo hi bars sigs =
  unzip [ (b, s) | (b, s) <- zip bars sigs, barDate b >= lo, null hi || barDate b < hi ]

-- backtest config with (k,β) on a date window, computing signals only there
run :: [Bar] -> (Field, Contrast) -> Double -> Int -> String -> String -> Report
run bars cfg beta k lo hi =
  let (bs, ss) = sliceCell lo hi bars (walkStencilRange cfg driftWin lookbackDays beta k lo hi bars)
  in backtest bs ss

bh :: [Bar] -> String -> String -> Report
bh bars lo hi = let (bs, _) = sliceCell lo hi bars (replicate (length bars) True) in buyHold bs

-- pick (k,β) maximising validation Sharpe over a β-set
selectArm :: [Bar] -> (Field, Contrast) -> [Double] -> String -> String -> (Int, Double)
selectArm bars cfg betas vlo vhi =
  snd $ maximum [ (repSharpe (run bars cfg b k vlo vhi), (k, b))
                | k <- kCandidates, b <- betas ]

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

evalConfig :: [Bar] -> (Field, Contrast) -> IO ()
evalConfig bars cfg = do
  printf "\n=== config %s ===\n" (configName cfg)
  printf "%-10s %-5s | %6s | %-14s | %-20s | %6s\n"
    "holdout" "reg" "B&H Sh" "beta=0 (Sh,ret)" "beta>0 (k,b,Sh,ret)" "delta"
  putStrLn (replicate 78 '-')
  rows <- forM windows $ \(vStart, hStart, hEnd) -> do
    let b0h        = bh bars hStart hEnd
        regimeBull = repTotalRet b0h > 0
        (k0, _)    = selectArm bars cfg [0.0]    vStart hStart
        (kB, bB)   = selectArm bars cfg betaPos  vStart hStart
        r0         = run bars cfg 0.0 k0 hStart hEnd
        rB         = run bars cfg bB  kB hStart hEnd
        delta      = repSharpe rB - repSharpe r0
    printf "%-10s %-5s | %+5.2f | k=%2d %+5.2f %+6.1f%% | k=%2d b=%.2f %+5.2f %+6.1f%% | %+5.2f\n"
      hStart (if regimeBull then "BULL" else "bear")
      (repSharpe b0h)
      k0 (repSharpe r0) (repTotalRet r0 * 100)
      kB bB (repSharpe rB) (repTotalRet rB * 100)
      delta
    return (regimeBull, repSharpe r0, repSharpe rB)
  let bulls = [ (s0, sB) | (True,  s0, sB) <- rows ]
      bears = [ (s0, sB) | (False, s0, sB) <- rows ]
      summ nm xs = printf "  %-5s windows: %d | mean beta=0 Sh %+5.2f | mean beta>0 Sh %+5.2f | mean delta %+5.2f | beta>0 wins %d/%d\n"
                     nm (length xs)
                     (mean (map fst xs)) (mean (map snd xs))
                     (mean (map (\(a,b) -> b - a) xs))
                     (length (filter (\(a,b) -> b > a) xs)) (length xs)
  putStrLn ""
  summ "BULL" bulls
  summ "bear" bears
  summ "ALL"  (bulls ++ bears)

-- Stitch the holdout slices into one continuous out-of-sample series,
-- each window using params selected on its own prior year.
stitch :: [Bar] -> (Field, Contrast) -> [Double] -> ([Bar], [Bool])
stitch bars cfg betas = unzip $ concat
  [ let (k, b)   = selectArm bars cfg betas vStart hStart
        (bs, ss) = sliceCell hStart hEnd bars
                     (walkStencilRange cfg driftWin lookbackDays b k hStart hEnd bars)
    in zip bs ss
  | (vStart, hStart, hEnd) <- windows ]

bottomLine :: [Bar] -> (Field, Contrast) -> IO ()
bottomLine bars cfg = do
  printf "\n=== bottom line, stitched out-of-sample (%s), %s onward ===\n"
    (configName cfg) (let (_, h, _) = head windows in h)
  let (sb, ssB) = stitch bars cfg betaPos        -- validated: reversion on
      (_,  ss0) = stitch bars cfg [0.0]          -- baseline: drift-only
      repB = backtest sb ssB
      rep0 = backtest sb ss0
      mkt  = buyHold sb
      line nm r = printf "  %-22s  ret %+8.1f%%  CAGR %+6.1f%%  Sharpe %5.2f  maxDD %6.1f%%  in-mkt %3.0f%%  trades %3d  ($1->$%.2f)\n"
                    nm (repTotalRet r*100) (repCAGR r*100) (repSharpe r) (repMaxDD r*100)
                    (repTimeIn r*100) (repTrades r) (repFinalMult r)
  printf "  (%d trading days, %s -> %s)\n" (length sb) (barDate (head sb)) (barDate (last sb))
  line "market (buy & hold)" mkt
  line "two-sided, beta=0"   rep0
  line "two-sided, reversion" repB

main :: IO ()
main = do
  csv <- readFile "data/PC1.csv"
  let bars = parseCSV csv
  printf "== multi-regime walk-forward on PC1 :: %d days %s -> %s ==\n"
    (length bars) (barDate (head bars)) (barDate (last bars))
  printf "lookback %d, driftWin %d, 6-month holdouts, validation = prior year\n" lookbackDays driftWin
  mapM_ (evalConfig bars) [(FP, Slope), (FP, Curv)]
  putStrLn "\nRead the BULL rows: if beta>0 holds up there too, m carries info beyond defensiveness."
  mapM_ (bottomLine bars) [(FP, Slope), (FP, Curv)]
