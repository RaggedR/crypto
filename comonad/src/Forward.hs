-- | The forward model: a linear equation of motion for log price, with
-- the momentum field as a forcing term, fit by ordinary least squares.
--
-- At each day we read three stencils off the tape of (log price p,
-- log momentum price m):
--
--   v = p_t - p_{t-1}              -- price velocity        [-1, 1]   (family a)
--   g = m_t - p_t                  -- gap / restoring force            (family b)
--   a = m_t - 2 m_{t-1} + m_{t-2}  -- momentum acceleration  [1,-2,1]  (family c)
--
-- and fit  Δp_{t+h} ≈ α·v + β·g + γ·a + c  by OLS. The acceleration
-- stencil is literally the Laplacian from the heat/wave equations in
-- tape.tex, now applied in time to the momentum field.
--
-- The coefficients are chosen by a convex, closed-form regression (no
-- grid search, nothing to overfit), so fishing risk lives only in the
-- single decision hyperparameter -- the horizon h -- which is validated
-- out of sample by the caller.
module Forward
  ( Row(..)
  , featureRows
  , fitOLS
  , predict
  , rSquared
  , forwardSignals
  , coinHash
  , isTrainCoin
  ) where

import Finance (Bar(..), momentumPrice)
import Data.Array
import Data.List (foldl')

-- | One training row: the day's features (with leading 1 for the
-- intercept) and the realised h-ahead log return. Tagged with the dates
-- of the feature day and the target day so the caller can split by time
-- without any look-ahead leakage.
data Row = Row
  { rowDate       :: !String   -- date of the feature day t
  , rowTargetDate :: !String   -- date of the target day t+h
  , rowX          :: ![Double] -- [1, v, g, a]
  , rowY          :: !Double   -- p_{t+h} - p_t
  }

-- | Per-day (log price, log momentum price), momentum undefined early on.
fields :: Int -> [Bar] -> (Array Int Bar, Int, Array Int (Maybe Double))
fields lookback bars = (barr, n, marr)
  where
    n    = length bars
    barr = listArray (0, n - 1) bars
    firstM = lookback
    marr = listArray (0, n - 1)
      [ if i < firstM
          then Nothing
          else case momentumPrice [ barr ! j | j <- [i - lookback .. i] ] of
                 Just mp | mp > 0 -> Just (log mp)   -- pool can go <=0 on crashes
                 _                -> Nothing
      | i <- [0 .. n - 1] ]

plog :: Array Int Bar -> Int -> Double
plog barr i = log (barClose (barr ! i))

-- | Build the feature rows for one coin. A row exists at day t when the
-- momentum field is defined at t, t-1, t-2 and the target day t+h is in
-- range. Rows are returned with no time filtering -- the caller decides
-- which belong to the training / validation / hold-out cells.
featureRows :: Int -> Int -> [Bar] -> [Row]
featureRows lookback horizon bars =
  [ Row (barDate (barr ! t)) (barDate (barr ! (t + horizon)))
        [1, v, g, a] y
  | t <- [lookback + 2 .. n - 1 - horizon]
  , Just mt  <- [marr ! t]
  , Just mt1 <- [marr ! (t - 1)]
  , Just mt2 <- [marr ! (t - 2)]
  , let v = plog barr t - plog barr (t - 1)
        g = mt - plog barr t
        a = mt - 2 * mt1 + mt2
        y = plog barr (t + horizon) - plog barr t
  , all ok [v, g, a, y]
  ]
  where
    (barr, n, marr) = fields lookback bars
    ok x = not (isNaN x || isInfinite x)

--------------------------------------------------------------------------
-- OLS (ridge) by normal equations + Gaussian elimination. k is tiny (4),
-- so this is trivial and dependency-free.
--------------------------------------------------------------------------

-- | Fit θ minimising Σ (y - x·θ)² + λ‖θ_{>0}‖² (intercept unregularised).
-- Accumulating XᵀX and Xᵀy is itself a fold over the rows.
fitOLS :: Double -> [Row] -> [Double]
fitOLS lambda rows = solve a b
  where
    k = case rows of (r:_) -> length (rowX r); [] -> 0
    z2 = replicate k (replicate k 0)
    z1 = replicate k 0
    (xtx, xty) = foldl' step (z2, z1) rows
    step (a0, b0) (Row _ _ x y) =
      ( zipWith (zipWith (+)) a0 [[xi * xj | xj <- x] | xi <- x]
      , zipWith (+) b0 (map (* y) x) )
    -- ridge on every coefficient except the intercept (index 0)
    a = [ [ xtx !! i !! j + (if i == j && i > 0 then lambda else 0)
          | j <- [0 .. k - 1] ] | i <- [0 .. k - 1] ]
    b = xty

-- | Solve A θ = b (A square) by Gaussian elimination with partial pivoting.
solve :: [[Double]] -> [Double] -> [Double]
solve a b = backSub (reduce (zipWith (\r y -> r ++ [y]) a b))
  where
    n = length b
    reduce m = foldl elim m [0 .. n - 1]
    elim m col =
      let piv = pivotFrom m col
          m'  = swap m col piv
          prow = m' !! col
          pel  = prow !! col
          norm = map (/ pel) prow
      in [ if r == col then norm
             else let row = m' !! r
                      f = row !! col
                  in zipWith (\ar nr -> ar - f * nr) row norm
         | r <- [0 .. n - 1] ]
    pivotFrom m col = snd $ maximum
      [ (abs (m !! r !! col), r) | r <- [col .. n - 1] ]
    swap m i j = [ if r == i then m !! j else if r == j then m !! i else m !! r
                 | r <- [0 .. length m - 1] ]
    backSub m = [ last (m !! r) | r <- [0 .. n - 1] ]

-- | Dot product θ · x.
predict :: [Double] -> [Double] -> Double
predict theta x = sum (zipWith (*) theta x)

-- | Coefficient of determination of θ on a row set (out-of-sample when
-- the rows are a hold-out cell).
rSquared :: [Double] -> [Row] -> Double
rSquared theta rows
  | null rows || sst == 0 = 0
  | otherwise             = 1 - ssr / sst
  where
    ys   = map rowY rows
    ybar = sum ys / fromIntegral (length ys)
    ssr  = sum [ (rowY r - predict theta (rowX r)) ^ (2 :: Int) | r <- rows ]
    sst  = sum [ (y - ybar) ^ (2 :: Int) | y <- ys ]

--------------------------------------------------------------------------
-- Tradeable signal: hold iff the fitted forward map forecasts a rise.
--------------------------------------------------------------------------

-- | One Bool per bar (chronological): hold iff θ·x > 0 at that day. Days
-- without defined features (early history) are cash. Uses only past
-- information (v, g, a are all causal), so this is Reading A -- tradeable.
--
-- Note the signal does not depend on the horizon h: h only shapes θ at
-- fit time (via the target). Given θ, the decision is a fixed linear
-- threshold on the current state.
forwardSignals :: [Double] -> Int -> [Bar] -> [Bool]
forwardSignals theta lookback bars =
  [ case marr ! t of
      _ | t < lookback + 2 -> False
      Nothing              -> False
      Just mt ->
        case (marr ! (t - 1), marr ! (t - 2)) of
          (Just mt1, Just mt2) ->
            let v = plog barr t - plog barr (t - 1)
                g = mt - plog barr t
                a = mt - 2 * mt1 + mt2
            in predict theta [1, v, g, a] > 0
          _ -> False
  | t <- [0 .. n - 1] ]
  where (barr, n, marr) = fields lookback bars

--------------------------------------------------------------------------
-- Deterministic coin split (so the train/test partition can't be
-- cherry-picked to flatter the result).
--------------------------------------------------------------------------

coinHash :: String -> Int
coinHash = foldl' (\h c -> (h * 131 + fromEnum c) `mod` 1000003) 7

-- | ~70% of coins to train, ~30% to test, by hash of the symbol.
isTrainCoin :: String -> Bool
isTrainCoin s = coinHash s `mod` 10 < 7
