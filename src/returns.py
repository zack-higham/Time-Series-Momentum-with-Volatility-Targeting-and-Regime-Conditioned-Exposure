"""Daily and monthly excess returns over the T-bill rate.

ETFs are fully funded, so their total return includes the cash rate. The
futures-based TSMOM literature works with excess returns (a futures position
earns the asset's return minus financing). Every return used for signals,
volatility and P&L in this project is therefore an excess return.

Monthly excess return for month t:
    (P_t / P_{t-1}) - prod_{days in t}(1 + rf_d)
where P is the total-return price at the last trading day of each month. This
is the return on a funded position minus the return on cash over the same
days, i.e. what a self-financed position earns. Summing daily excess returns
instead would ignore compounding.
"""

import pandas as pd

from config import INSTRUMENTS, SAMPLE_END
from data_io import load_frame
from riskfree import daily_rf


def load_prices() -> pd.DataFrame:
    """Total-return (distribution-adjusted) daily prices, in config order."""
    return load_frame("adj_close").loc[:SAMPLE_END, list(INSTRUMENTS)]


def load_prices_u2() -> pd.DataFrame:
    """Phase 2 universe: the phase 1 files for the 10 core ETFs plus the 24
    additions, on the same trading calendar, in config order."""
    from config import UNIVERSE_U2
    core = load_frame("adj_close").loc[:SAMPLE_END]
    add = load_frame("u2add_adj_close").loc[:SAMPLE_END]
    assert core.index.equals(add.index), "phase 1 and phase 2 calendars differ"
    return pd.concat([core, add], axis=1)[list(UNIVERSE_U2)]


def month_end_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month."""
    s = trading_days.to_series()
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # fill_method=None: a missing price must give a missing return, never a
    # return computed against a stale, forward-filled price.
    return prices.pct_change(fill_method=None)


def daily_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rf = daily_rf(prices.index)
    return daily_returns(prices).sub(rf, axis=0)


def monthly_rf(trading_days: pd.DatetimeIndex) -> pd.Series:
    """Compounded cash return over each month, indexed by its last trading day."""
    rf = daily_rf(trading_days).fillna(0.0)
    growth = (1 + rf).groupby(trading_days.to_period("M")).prod() - 1
    growth.index = month_end_dates(trading_days)
    return growth.rename("rf")


def monthly_excess_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end-to-month-end total return minus compounded cash.

    A ticker's first, partial month is NaN because the previous month-end
    price does not exist.
    """
    ends = month_end_dates(prices.index)
    total = prices.loc[ends].pct_change(fill_method=None)
    return total.sub(monthly_rf(prices.index), axis=0)
