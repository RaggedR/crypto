-- | A configurable two-sided stencil strategy, for tuning on PC1.
--
-- A stencil is parameterised by three knobs (kept deliberately few, per
-- the PCA: ~2.5 effective factors => tiny honest tuning budget):
--
--   * Field    — which series it reads: price p, momentum m, or gap g=m-p
--   * Contrast — slope (centred 1st difference) or curvature ([1,-2,1])
--   * k        — the half-window / stencil arm
--
-- Hold iff the contrast is positive: for Slope that is "future side above
-- past side" (zero-lag momentum); for Curv that is convex/trough
-- ("buy the dip" made precise). The future side is supplied by the
-- forward map ('Boundary.forecastFrom') so the rule fires at the edge.
module Stencil
  ( Field(..), Contrast(..)
  , walkStencil, walkStencilRange, edgeStencil
  , allConfigs, configName
  ) where

import Finance (Bar(..), momentumPrice)
import Boundary (forecastReversion)
import Data.Array

data Field    = FP | FM | FG            deriving (Eq, Enum, Bounded)
data Contrast = Slope | Curv            deriving (Eq, Enum, Bounded)

allConfigs :: [(Field, Contrast)]
allConfigs = [ (f, c) | f <- [minBound ..], c <- [minBound ..] ]

configName :: (Field, Contrast) -> String
configName (f, c) = fld f ++ "/" ++ con c
  where fld FP = "p"; fld FM = "m"; fld FG = "g"
        con Slope = "slope"; con Curv = "curv"

mean :: [Double] -> Double
mean [] = 0
mean xs = sum xs / fromIntegral (length xs)

-- field value at an index of a bar array (momentum needs a causal window)
fieldAt :: Field -> Int -> Array Int Bar -> Int -> Maybe Double
fieldAt FP _  arr i = Just (log (barClose (arr ! i)))
fieldAt FM lb arr i = mLogAt lb arr i
fieldAt FG lb arr i = (\m -> m - log (barClose (arr ! i))) <$> mLogAt lb arr i

mLogAt :: Int -> Array Int Bar -> Int -> Maybe Double
mLogAt lb arr i
  | i < lb    = Nothing
  | otherwise = case momentumPrice [ arr ! j | j <- [i - lb .. i] ] of
                  Just mp | mp > 0 -> Just (log mp)
                  _                -> Nothing

-- the contrast at focus t, reading the field through a bounds-safe lookup
contrastAt :: Contrast -> Int -> (Int -> Maybe Double) -> Int -> Maybe Double
contrastAt Slope k f t = do
  rs <- mapM f [t + 1 .. t + k]        -- future side (forecast at the edge)
  ls <- mapM f [t - k .. t - 1]        -- past side
  return (mean rs - mean ls)
contrastAt Curv k f t = do
  fr <- f (t + k); fc <- f t; fl <- f (t - k)
  return (fr - 2 * fc + fl)            -- [1,-2,1] with arm k; >0 = trough

safeLookup :: Field -> Int -> Array Int Bar -> Int -> Int -> Maybe Double
safeLookup fld lb arr hi i
  | i < 0 || i > hi = Nothing
  | otherwise       = fieldAt fld lb arr i

decide :: (Field, Contrast) -> Int -> Int -> Array Int Bar -> Int -> Int -> Bool
decide (fld, con) k lb arr hi t =
  case contrastAt con k (safeLookup fld lb arr hi) t of
    Just c  -> c > 0
    Nothing -> False

--------------------------------------------------------------------------

-- | HONEST (live-reproducible): at each day forecast the right side from
-- data up to that day only (drift + reversion-to-@m@ at rate β), then
-- evaluate the stencil. One Bool per bar.
walkStencil :: (Field, Contrast) -> Int -> Int -> Double -> Int -> [Bar] -> [Bool]
walkStencil cfg driftWin lb beta k bars =
  [ let hist = take (t + 1) bars
        ext  = hist ++ forecastReversion driftWin lb k beta (reverse hist)
        hi   = t + k
        arr  = listArray (0, hi) ext
    in decide cfg k lb arr hi t
  | t <- [0 .. n - 1] ]
  where n = length bars

-- | As 'walkStencil', but only compute the (expensive) signal for days in
-- the date window [lo, hi); others are cash. Lets a walk-forward driver
-- evaluate many windows without recomputing the whole 2000+ day series.
walkStencilRange :: (Field, Contrast) -> Int -> Int -> Double -> Int
                 -> String -> String -> [Bar] -> [Bool]
walkStencilRange cfg driftWin lb beta k lo hi bars =
  [ if inWin (barDate bar)
      then let hist = take (t + 1) bars
               ext  = hist ++ forecastReversion driftWin lb k beta (reverse hist)
               top  = t + k
               arr  = listArray (0, top) ext
           in decide cfg k lb arr top t
      else False
  | (t, bar) <- zip [0 ..] bars ]
  where inWin d = d >= lo && (null hi || d < hi)

-- | IDEAL (look-ahead in the interior): append the forecast once at the
-- end, then evaluate the stencil at every real day. Interior days read
-- the real future; only the last k read the forecast.
edgeStencil :: (Field, Contrast) -> Int -> Int -> Double -> Int -> [Bar] -> [Bool]
edgeStencil cfg driftWin lb beta k bars =
  [ decide cfg k lb arr hi t | t <- [0 .. n - 1] ]
  where
    n   = length bars
    ext = bars ++ forecastReversion driftWin lb k beta (reverse bars)
    hi  = length ext - 1
    arr = listArray (0, hi) ext
