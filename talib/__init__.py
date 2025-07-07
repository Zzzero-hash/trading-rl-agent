import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

def RSI(close, timeperiod=14):
    series = pd.Series(close, dtype=float)
    return RSIIndicator(series, window=timeperiod).rsi().to_numpy()

def _zeros(*args):
    n = len(args[0])
    return np.zeros(n, dtype=int)

CDLDOJI = _zeros
CDLHAMMER = _zeros
CDLENGULFING = _zeros
CDLSHOOTINGSTAR = _zeros
CDLMORNINGSTAR = _zeros
CDLEVENINGSTAR = _zeros
