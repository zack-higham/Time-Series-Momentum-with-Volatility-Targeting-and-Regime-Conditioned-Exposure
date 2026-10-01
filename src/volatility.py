"""Ex-ante volatility, correlation and covariance estimates.

Volatility follows Moskowitz, Ooi and Pedersen (2012): an exponentially
weighted variance of daily excess returns around their exponentially weighted
mean, with centre of mass 60 days, annualised with 252 trading days:

    sigma_t^2 = 252 * sum_i (1 - d) d^i (r_{t-i} - rbar_t)^2,   d / (1 - d) = 60.

pandas' ewm(com=60) uses exactly this decay, d = 60/61. The estimate dated t
uses returns up to and including day t, i.e. it is known at the close of t.
Correlations use a trailing window of daily excess returns ending at t.
"""

import numpy as np
import pandas as pd

from config import (ANNUALISATION_DAYS, CORR_MIN_OBS, CORR_WINDOW_DAYS, EWMA_COM_DAYS,
                    VOL_MIN_OBS)


def ewma_vol(daily_excess: pd.DataFrame, com: float = EWMA_COM_DAYS,
             min_obs: int = VOL_MIN_OBS) -> pd.DataFrame:
    """Annualised EWMA volatility for every ticker and day.

    Computed per column on that ticker's own valid history, so leading NaNs
    before inception do not affect the weights.
    """
    out = {}
    for t in daily_excess.columns:
        r = daily_excess[t].dropna()
        # bias=True: weights sum to one, matching the MOP formula (no
        # small-sample correction).
        var = r.ewm(com=com, min_periods=min_obs).var(bias=True)
        out[t] = np.sqrt(var * ANNUALISATION_DAYS)
    return pd.DataFrame(out).reindex(daily_excess.index)


def trailing_correlation(daily_excess: pd.DataFrame, date: pd.Timestamp,
                         window: int = CORR_WINDOW_DAYS,
                         min_obs: int = CORR_MIN_OBS) -> pd.DataFrame:
    """Pairwise correlation of daily excess returns over the window ending at date."""
    hist = daily_excess.loc[:date].tail(window)
    return hist.corr(min_periods=min_obs)


def covariance(vols: pd.Series, corr: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance D C D from vols and a correlation matrix."""
    d = np.diag(vols.values)
    return pd.DataFrame(d @ corr.loc[vols.index, vols.index].values @ d,
                        index=vols.index, columns=vols.index)
