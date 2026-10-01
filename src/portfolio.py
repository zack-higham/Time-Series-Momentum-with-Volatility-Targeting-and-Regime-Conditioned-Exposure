"""Target weights for each strategy variant, decided at month-end signal dates."""

import pandas as pd

from config import MAIN_START, SAMPLE_END
from returns import month_end_dates


def rebalance_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Signal dates whose holding month lies inside the main sample.

    The first is the month-end before MAIN_START (positions for April 2008 are
    set at the close of 2008-03-31). The last is the month-end before
    SAMPLE_END: a signal at the final month-end would be for a month outside
    the sample. Every date is a complete month-end (the last trading day of
    its calendar month), never a partial month.
    """
    ends = month_end_dates(trading_days)
    last_day = pd.Timestamp(SAMPLE_END)
    assert last_day in ends, "sample must end on a complete month-end"
    first_holding = pd.Period(MAIN_START, "M")
    dates = ends[(ends.to_period("M") >= first_holding - 1) & (ends < last_day)]
    return dates


def raw_tsmom_weights(signal: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Variant 1: equal notional per instrument, w_i = s_i / N (gross exposure <= 1).

    N counts the instruments with a signal on that date; in the main sample all
    ten always have one (asserted by the caller).
    """
    s = signal.loc[dates]
    n = s.notna().sum(axis=1)
    return s.div(n, axis=0)
