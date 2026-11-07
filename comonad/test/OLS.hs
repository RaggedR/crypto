-- | Sanity check for the hand-rolled OLS / Gaussian elimination: generate
-- rows from a known θ with a little deterministic noise, fit, and confirm
-- we recover the coefficients. If this passes, the forward-model fit can
-- be trusted.
module Main (main) where

import Forward (Row(..), fitOLS, predict, rSquared)
import System.Exit (exitFailure)
import Text.Printf (printf)

-- deterministic pseudo-noise in [-0.5,0.5], no Math.random needed
noise :: Int -> Double
noise i = (fromIntegral ((i * 2654435761) `mod` 1000) / 1000) - 0.5

main :: IO ()
main = do
  let trueTheta = [0.3, -1.2, 0.8, 2.5]      -- intercept, v, g, a
      rows =
        [ Row (show i) (show i) x y
        | i <- [0 .. 4000]
        , let v = noise i
              g = noise (i + 7)
              a = noise (i + 13)
              x = [1, v, g, a]
              y = predict trueTheta x + 0.01 * noise (i + 101)  -- small noise
        ]
      est = fitOLS 0 rows
      r2  = rSquared est rows
      errs = zipWith (\a b -> abs (a - b)) est trueTheta
      ok   = all (< 0.02) errs && r2 > 0.99

  printf "true  θ = %s\n" (show trueTheta)
  printf "fitted θ = %s\n" (show [ fromIntegral (round (x * 1000) :: Int) / 1000 :: Double | x <- est ])
  printf "max coef error = %.4f   R² = %.5f\n" (maximum errs) r2
  if ok
    then putStrLn "OLS RECOVERS KNOWN COEFFICIENTS"
    else putStrLn "OLS FIT WRONG" >> exitFailure
