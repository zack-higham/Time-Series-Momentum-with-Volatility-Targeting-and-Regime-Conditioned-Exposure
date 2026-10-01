"""Daily risk-free return from the ^IRX T-bill rate.

^IRX is an annualised percentage rate. The return earned from the close of
trading day t-1 to the close of day t is the rate known at t-1, accrued over
the calendar days in between on an actual/360 basis (the money-market
convention for T-bills), so a Friday-to-Monday period earns three days.
Days when the bond market is shut but ETFs trade (Columbus Day, Veterans Day)
carry the previous rate forward: it is the latest rate known, so this
introduces no lookahead.
"""

import pandas as pd

from data_io import load_frame


def daily_rf(trading_days: pd.DatetimeIndex) -> pd.Series:
    rate = load_frame("irx")["irx_pct"].reindex(trading_days).ffill() / 100.0
    days = trading_days.to_series().diff().dt.days
    rf = rate.shift(1) * days / 360.0
    return rf.rename("rf")
