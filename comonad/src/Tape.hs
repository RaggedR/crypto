{-# LANGUAGE DeriveFunctor #-}

-- | The Tape and Stream comonads.
--
-- Transcribed from Robin's note @aladdin/tape.tex@. The Tape is the
-- bi-infinite, two-sided comonad: a focus with an infinite run of values
-- to each side, so a local rule can see left (past) /and/ right (future).
-- The Stream is its one-sided, causal cousin: a focus with an infinite
-- run to the right only, so a local rule can see the present and the
-- past but never the future.
--
-- The handedness is the whole point. A *live, tradeable* rule must be a
-- Stream co-Kleisli arrow -- reading tomorrow's price is physically
-- impossible. The Tape is the *analysis* object: we can only read its
-- right-hand side once that side exists (a backtest) or once we have
-- forecast it (a model).
module Tape
  ( Comonad(..)
  , extend
  , (=<=)
  , cokPow
    -- * Two-sided: the tape
  , Tape(..)
  , leftT
  , rightT
  , fromListT
  , takeR
  , takeL
    -- * One-sided: the causal stream
  , Stream(..)
  , fromListS
  , takeS
  ) where

-- | A comonad: every value sits in a context, @extract@ reads the value
-- \"here\", @duplicate@ exposes every way of re-focusing.
class Functor w => Comonad w where
  extract   :: w a -> a           -- ^ the value under the focus
  duplicate :: w a -> w (w a)     -- ^ every re-focusing

-- | Run a local rule at every position. This is the backtest engine:
-- @extend strategy@ applies @strategy@ at every day of history at once.
extend :: Comonad w => (w a -> b) -> w a -> w b
extend f = fmap f . duplicate

-- | Co-Kleisli composition: @g@ after @f@.
(=<=) :: Comonad w => (w b -> c) -> (w a -> b) -> (w a -> c)
g =<= f = g . extend f
infixr 1 =<=

-- | The n-th co-Kleisli power of a local step.
cokPow :: Comonad w => Int -> (w a -> a) -> (w a -> a)
cokPow 0 _ = extract
cokPow n f = f =<= cokPow (n - 1) f

--------------------------------------------------------------------------
-- The bi-infinite tape (two-sided): sees past and future.
--------------------------------------------------------------------------

-- | Lefts (nearest first), focus, rights (nearest first).
data Tape a = Tape [a] a [a] deriving (Functor)

leftT, rightT :: Tape a -> Tape a
leftT  (Tape (l:ls) x rs) = Tape ls l (x:rs)
leftT  (Tape []     x rs) = Tape [] x rs          -- clamp at a finite edge
rightT (Tape ls x (r:rs)) = Tape (x:ls) r rs
rightT (Tape ls x [])     = Tape ls x []          -- clamp at a finite edge

instance Comonad Tape where
  extract (Tape _ x _) = x
  duplicate t = Tape (tail (iterate leftT  t))
                     t
                     (tail (iterate rightT t))

-- | A finite signal becomes a tape by padding the left with the given
-- pad value; the right runs off into the rest of the list. For a true
-- bi-infinite tape pad with @repeat p@ by using 'Tape' directly.
fromListT :: a -> [a] -> Tape a
fromListT p (x:xs) = Tape (repeat p) x xs
fromListT p []     = Tape (repeat p) p (repeat p)

-- | Read @k@ values from the focus rightward (focus included).
takeR :: Int -> Tape a -> [a]
takeR k (Tape _ x rs) = x : take (k - 1) rs

-- | Read @k@ values from the focus leftward (focus included, nearest first).
takeL :: Int -> Tape a -> [a]
takeL k (Tape ls x _) = x : take (k - 1) ls

--------------------------------------------------------------------------
-- The one-sided stream (causal): sees only present and past.
--------------------------------------------------------------------------

-- | Past values (nearest first) and the focus. There is no \"right\":
-- a Stream rule structurally /cannot/ read the future. This is the
-- type of a live, tradeable strategy.
data Stream a = Stream [a] a deriving (Functor)

-- shift the focus one step into the past
pastS :: Stream a -> Stream a
pastS (Stream (p:ps) _) = Stream ps p
pastS s@(Stream [] _)   = s                        -- clamp at the beginning

instance Comonad Stream where
  extract (Stream _ x) = x
  duplicate s = Stream (tail (iterate pastS s)) s

-- | Build a causal stream whose focus is the /last/ element of the list
-- (the present), with everything before it as the past (nearest first).
fromListS :: [a] -> Stream a
fromListS [] = error "fromListS: empty history"
fromListS xs = Stream (reverse (init xs)) (last xs)

-- | Read @k@ values back from the focus (focus included, nearest first).
takeS :: Int -> Stream a -> [a]
takeS k (Stream ps x) = x : take (k - 1) ps
