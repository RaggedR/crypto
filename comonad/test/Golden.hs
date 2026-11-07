{-# LANGUAGE ScopedTypeVariables #-}

-- | Correctness anchor for the comonad core.
--
-- Reproduces the numeric traces printed in @aladdin/tape.tex@: the
-- Gaussian smoothing of an impulse (§1) and the d'Alembert splitting of
-- a bump under the wave equation (§3). If these match, @extend@,
-- @duplicate@ and co-Kleisli powers are behaving exactly as the note
-- claims, and the finance code can trust them.
module Main (main) where

import Tape
import System.Exit (exitFailure)
import Text.Printf (printf)

-- A bi-infinite tape from a centre value and symmetric neighbours.
mkTape :: Num a => a -> [a] -> Tape a
mkTape x sides = Tape (sides ++ repeat 0) x (sides ++ repeat 0)

-- §1: the centred Gaussian kernel [1,2,1]/4.
gauss3 :: Fractional a => Tape a -> a
gauss3 (Tape (l:_) x (r:_)) = 0.25 * l + 0.5 * x + 0.25 * r
gauss3 _                    = error "gauss3: need both neighbours"

smooth :: Fractional a => Int -> Tape a -> Tape a
smooth n = extend (cokPow n gauss3)

-- Read a symmetric window (focus boxed) around the centre for display.
windowT :: Int -> Tape a -> ([a], a, [a])
windowT k (Tape ls x rs) = (reverse (take k ls), x, take k rs)

-- §3: the leapfrog wave step over Tape (current, previous).
wave :: Fractional a => Tape (a, a) -> (a, a)
wave (Tape ((left,_):_) (cur,prev) ((right,_):_)) = (new, cur)
  where lap = left - 2*cur + right
        new = 2*cur - prev + lap
wave _ = error "wave: need both neighbours"

evolve :: Fractional a => Int -> Tape (a, a) -> Tape (a, a)
evolve n = extend (cokPow n wave)

approx :: Double -> Double -> Bool
approx a b = abs (a - b) < 1e-9

-- Compare a computed symmetric window against the note's printed row.
checkRow :: String -> Int -> Tape Double -> [Double] -> IO Bool
checkRow label k t expected = do
  let (ls, x, rs) = windowT k t      -- windowT already returns ls farthest-first
      got = ls ++ [x] ++ rs
      ok  = length got == length expected && and (zipWith approx got expected)
  printf "%-10s %s  %s\n" label
    (unwords (map (printf "%6.2f") got))
    (if ok then "ok" else "MISMATCH expected " ++ unwords (map (printf "%.2f") expected))
  return ok

main :: IO ()
main = do
  putStrLn "== Tape comonad golden traces (from tape.tex) =="
  putStrLn "\n-- §1 heat/Gaussian: smoothing of the impulse 16"
  let spike = mkTape 16 []
      -- expected rows, 4 cells each side of the boxed focus
      heatExpected =
        [ ("step 0", [0,0,0,0, 16, 0,0,0,0])
        , ("step 1", [0,0,0,4,  8, 4,0,0,0])
        , ("step 2", [0,0,1,4,  6, 4,1,0,0])
        , ("step 3", [0,0.25,1.5,3.75, 5, 3.75,1.5,0.25,0])
        , ("step 4", [0.0625,0.5,1.75,3.5, 4.375, 3.5,1.75,0.5,0.0625])
        ]
  heatOks <- mapM (\(i,(lbl,ex)) -> checkRow lbl 4 (smooth i spike) ex)
                  (zip [0..] heatExpected)

  putStrLn "\n-- §3 wave: a bump at rest splits into two travelling halves"
  let bump = Tape ([2,1] ++ repeat 0) 3 ([2,1] ++ repeat 0)
      atRest (Tape (l:_) x (r:_)) = (x + (l - 2*x + r)/2, x)
      atRest _ = error "atRest"
      seed :: Tape (Double, Double)
      seed = extend atRest bump
      disp k = fmap fst (evolve k seed)
      waveExpected =
        [ ("frame 0", [0,0,0.5,1,2, 2, 2,1,0.5,0,0])
        , ("frame 1", [0,0.5,1,1.5,1, 1, 1,1.5,1,0.5,0])
        , ("frame 2", [0.5,1,1.5,1,0.5, 0, 0.5,1,1.5,1,0.5])
        , ("frame 3", [1,1.5,1,0.5,0, 0, 0,0.5,1,1.5,1])
        , ("frame 4", [1.5,1,0.5,0,0, 0, 0,0,0.5,1,1.5])
        ]
  waveOks <- mapM (\(i,(lbl,ex)) -> checkRow lbl 5 (disp i) ex)
                  (zip [0..] waveExpected)

  let allOk = and (heatOks ++ waveOks)
  putStrLn ""
  if allOk
    then putStrLn "ALL GOLDEN TRACES MATCH tape.tex"
    else putStrLn "GOLDEN TRACE MISMATCH" >> exitFailure
