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


def period_excess(daily_excess: pd.Series, rf: pd.Series, freq: str) -> pd.Series:
    """Compound daily excess returns to calendar periods ("M" or "Q"):
    prod(1 + rf + R) - prod(1 + rf), the same definition as monthly_excess."""
    df = pd.DataFrame({"R": daily_excess, "rf": rf.reindex(daily_excess.index).fillna(0.0)}).dropna()
    by = df.index.to_period(freq)
    return (1 + df["rf"] + df["R"]).groupby(by).prod() - (1 + df["rf"]).groupby(by).prod()


def lo_sharpe_se(monthly: pd.Series) -> float:
    """Lo (2002) IID standard error of the annualised Sharpe ratio:
    sqrt((1 + SR_m^2 / 2) / T) per month, times sqrt(12)."""
    m = monthly.dropna()
    sr = m.mean() / m.std()
    return float(np.sqrt((1 + 0.5 * sr ** 2) / len(m)) * np.sqrt(12))


def sortino(monthly: pd.Series) -> float:
    """Annualised mean over annualised downside deviation (target 0)."""
    m = monthly.dropna()
    downside = np.sqrt(np.mean(np.minimum(m, 0.0) ** 2)) * np.sqrt(12)
    return float(m.mean() * 12 / downside)


def drawdown_episode(daily_excess: pd.Series) -> dict:
    """Deepest drawdown: peak, trough, recovery dates and durations in months
    (calendar days / 30.44). The longest underwater spell is also returned."""
    wealth = (1 + daily_excess.dropna()).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery = recovered.index[0] if len(recovered) else None
    # Longest spell between successive new highs (open spell runs to the end).
    highs = dd.index[dd >= 0].append(pd.DatetimeIndex([dd.index[-1]]))
    gaps = highs.to_series().diff().dt.days
    return {"mdd_peak": peak.date(), "mdd_trough": trough.date(),
            "mdd_recovery": recovery.date() if recovery is not None else None,
            "mdd_peak_to_trough_m": (trough - peak).days / 30.44,
            "mdd_underwater_m": ((recovery or dd.index[-1]) - peak).days / 30.44,
            "longest_underwater_m": gaps.max() / 30.44}


def moments(x: pd.Series) -> dict:
    """Skewness and excess kurtosis with normal-theory standard errors,
    sqrt(6/n) and sqrt(24/n) (understated for fat-tailed data)."""
    x = x.dropna()
    n = len(x)
    return {"skew": x.skew(), "skew_se": np.sqrt(6 / n),
            "exkurt": x.kurt(), "exkurt_se": np.sqrt(24 / n), "n": n}
