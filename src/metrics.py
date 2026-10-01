"""Performance statistics for monthly and daily excess-return series."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import newey_west_lags


def nw_tstat(x: pd.Series) -> float:
    """Newey-West t-statistic of the mean (automatic lag rule)."""
    x = x.dropna()
    fit = sm.OLS(x.values, np.ones(len(x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(x))})
    return float(fit.tvalues[0])


def max_drawdown(daily_excess: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative excess-return index."""
    wealth = (1 + daily_excess.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(monthly: pd.Series, daily: pd.Series) -> dict:
    """Core statistics. Annualised mean and volatility from monthly returns."""
    m = monthly.dropna()
    ann_ret = m.mean() * 12
    ann_vol = m.std() * np.sqrt(12)
    mdd = max_drawdown(daily)
    return {
        "ann_ret_pct": ann_ret * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": ann_ret / ann_vol,
        "nw_t": nw_tstat(m),
        "max_dd_pct": mdd * 100,
        "calmar": ann_ret / abs(mdd),
        "skew_m": m.skew(),
        "hit_rate": (m > 0).mean(),
        "months": len(m),
    }
