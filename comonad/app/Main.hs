-- | Run the three readings of the bookmaker model over one asset and
-- print a comparison: causal (live), forecast (Reading A, tradeable),
-- oracle (Reading B, perfect-foresight ceiling), and buy-and-hold.
module Main (main) where

import Finance
import System.Environment (getArgs)
import Text.Printf (printf)

main :: IO ()
main = do
  args <- getArgs
  let (path, lookback, horizon) = case args of
        (p:l:h:_) -> (p, read l, read h)
        (p:l:_)   -> (p, read l, 20)
        [p]       -> (p, 180, 20)
        []        -> ("data/BTCUSDT.csv", 180, 20)

  csv <- readFile path
  let bars = parseCSV csv
      n    = length bars

  printf "== %s : %d daily bars, %s -> %s ==\n"
    path n (barDate (head bars)) (barDate (last bars))
  printf "lookback = %d days, forecast horizon = %d days\n\n" (lookback :: Int) (horizon :: Int)

  let causal   = extendS (causalSignal lookback)          bars
      forecast = extendS (forecastSignal lookback horizon) bars
      oracle   = extendT  oracleSignal                     bars

  putStrLn (formatReport "buy & hold"  (buyHold bars))
  putStrLn (formatReport "causal (live)" (backtest bars causal))
  putStrLn (formatReport "forecast (A)"  (backtest bars forecast))
  putStrLn (formatReport "oracle (B)"    (backtest bars oracle))
  putStrLn ""
  putStrLn "causal & forecast are tradeable; oracle is the perfect-foresight ceiling only."
