"""Time-series momentum signals.

For lookback k months and month-end t, the signal is the sign of the
instrument's cumulative excess return over months t-k+1, ..., t:

    s_t^k = sign( P_t / P_{t-k} - prod(1 + rf) over the same k months ).

The signal is known at the close of month-end t and is used for the position
held over month t+1. Following MOP (2012) there is no skipped month: in time
series momentum the most recent month is part of the trend, unlike the
cross-sectional "12-1" convention used in Project 1.
"""

import numpy as np
import pandas as pd

from config import LOOKBACKS_MONTHS
from returns import month_end_dates, monthly_rf


def cumulative_excess_return(prices: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-month cumulative excess return at every month-end (NaN without full history)."""
    ends = month_end_dates(prices.index)
    p = prices.loc[ends]
    total = p / p.shift(k) - 1
    cash = (1 + monthly_rf(prices.index)).rolling(k).apply(np.prod, raw=True) - 1
    return total.sub(cash, axis=0)


def tsmom_signal(prices: pd.DataFrame, k: int) -> pd.DataFrame:
    """+1 / -1 signal; NaN where the k-month history is incomplete.

    An exactly zero cumulative excess return (never observed in practice) maps
    to 0, i.e. no position.
    """
    return np.sign(cumulative_excess_return(prices, k))


def all_signals(prices: pd.DataFrame) -> dict:
    """Signals for each lookback plus the equal-weight blend of the four.

    The blend is defined only when every component is available, so it never
    silently averages fewer horizons.
    """
    signals = {k: tsmom_signal(prices, k) for k in LOOKBACKS_MONTHS}
    stacked = np.stack([signals[k].values for k in LOOKBACKS_MONTHS])
    blend = np.mean(stacked, axis=0)  # NaN if any component is NaN
    signals["blend"] = pd.DataFrame(blend, index=signals[12].index,
                                    columns=signals[12].columns)
    return signals
