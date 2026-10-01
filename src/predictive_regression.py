"""Pooled predictive regressions and the TSMOM return decomposition.

Regressions follow MOP (2012, Section 3). Returns are scaled by ex-ante
volatility so that instruments with very different volatilities (IEF at ~7%,
EEM at ~28%) are comparable in one pooled regression:

    z_{i,t} = r_{i,t} / sigma_{i,t-1}

where r is the monthly excess return and sigma the annualised EWMA volatility
known at the end of month t-1. Two families:
  (a) lag regressions:    z_{i,t} = a + b_h z_{i,t-h} + e     (h = 1..24)
  (b) signal regressions: z_{i,t} = a + b_k s^k_{i,t-1} + e   (k = 1,3,6,12)
Standard errors are clustered by month, because the same macro shock hits
every instrument in a given month, so observations within a month are not
independent. Each regression is also run with instrument fixed effects
(within-instrument demeaning), which removes the part of the pooled
coefficient that comes from instruments with different average returns
rather than from genuine predictability (Huang, Li, Wang and Zhou 2020).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def scaled_returns(monthly_excess: pd.DataFrame, month_end_vol: pd.DataFrame) -> pd.DataFrame:
    """z_{i,t} = r_{i,t} / sigma_{i,t-1}: next month's return over the vol known beforehand."""
    return monthly_excess / month_end_vol.shift(1)


def _pooled_ols(y: pd.DataFrame, x: pd.DataFrame, months: pd.DatetimeIndex,
                fixed_effects: bool) -> dict:
    """Stack two (month x instrument) panels and run pooled OLS clustered by month."""
    panel = pd.DataFrame({"y": y.loc[months].stack(), "x": x.loc[months].stack()}).dropna()
    if fixed_effects:
        by_inst = panel.groupby(level=1)
        panel = panel - by_inst.transform("mean")
        exog = panel[["x"]]
    else:
        exog = sm.add_constant(panel[["x"]])
    month_id = pd.factorize(panel.index.get_level_values(0))[0]
    fit = sm.OLS(panel["y"], exog).fit(cov_type="cluster", cov_kwds={"groups": month_id})
    return {"coef": fit.params["x"], "t": fit.tvalues["x"], "n_obs": int(fit.nobs),
            "n_months": panel.index.get_level_values(0).nunique()}


def lag_regressions(z: pd.DataFrame, months: pd.DatetimeIndex, max_lag: int) -> pd.DataFrame:
    """(a) z_t on z_{t-h} and on sign(r_{t-h}) for each lag h, pooled and with fixed effects."""
    rows = []
    for h in range(1, max_lag + 1):
        lagged = z.shift(h)
        for regressor, x in [("return", lagged), ("sign", np.sign(lagged))]:
            for fe in (False, True):
                res = _pooled_ols(z, x, months, fe)
                rows.append({"lag": h, "regressor": regressor, "fixed_effects": fe, **res})
    return pd.DataFrame(rows)


def signal_regressions(z: pd.DataFrame, signals: dict, months: pd.DatetimeIndex) -> pd.DataFrame:
    """(b) z_t on the k-month signal known at t-1, pooled and with fixed effects."""
    rows = []
    for k, s in signals.items():
        for fe in (False, True):
            res = _pooled_ols(z, s.shift(1), months, fe)
            rows.append({"lookback": k, "fixed_effects": fe, **res})
    return pd.DataFrame(rows)


def per_instrument_signal_regressions(z: pd.DataFrame, signal: pd.DataFrame,
                                      months: pd.DatetimeIndex, nw_lags: int) -> pd.DataFrame:
    """Time-series regression of z_t on s_{t-1} for each instrument (Newey-West t)."""
    rows = []
    for t in z.columns:
        df = pd.DataFrame({"y": z.loc[months, t], "x": signal[t].shift(1).loc[months]}).dropna()
        fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        rows.append({"ticker": t, "coef": fit.params["x"], "t": fit.tvalues["x"],
                     "frac_long": (df["x"] > 0).mean(), "n": int(fit.nobs)})
    return pd.DataFrame(rows).set_index("ticker")


def decompose(z: pd.DataFrame, x: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Split each instrument's average strategy return E[x_{t-1} z_t] into
    Cov(x_{t-1}, z_t), the timing (autocovariance) part, and E[x] E[z], the
    part earned simply because the position is on average long (or short) an
    asset with a non-zero mean return. Population moments, so the two parts
    add up exactly.
    """
    rows = []
    xs = x.shift(1)
    for t in z.columns:
        df = pd.DataFrame({"x": xs.loc[months, t], "y": z.loc[months, t]}).dropna()
        total = (df["x"] * df["y"]).mean()
        mean_part = df["x"].mean() * df["y"].mean()
        rows.append({"ticker": t, "total": total, "timing": total - mean_part,
                     "static": mean_part, "mean_position": df["x"].mean(),
                     "mean_z": df["y"].mean(), "n": len(df)})
    return pd.DataFrame(rows).set_index("ticker")
