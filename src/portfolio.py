"""Target weights for each strategy variant, decided at month-end signal dates."""

import numpy as np
import pandas as pd

from config import INSTRUMENTS, MAIN_START, PORTFOLIO_VOL_TARGET, SAMPLE_END
from returns import month_end_dates
from volatility import covariance, trailing_correlation


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


def risk_budgets(allocation: str, exclude_class: str | None = None,
                 classes: dict | None = None) -> pd.Series:
    """Share of ex-ante risk budget per instrument, summing to one.

    "asset_class": 25% per asset class, split equally within the class.
    "instrument":  1/N per instrument (MOP convention).
    exclude_class drops one asset class (leave-one-out robustness); the
    remaining classes share the budget equally.
    """
    classes = pd.Series(INSTRUMENTS if classes is None else classes)
    if exclude_class is not None:
        assert exclude_class in classes.values, f"unknown class {exclude_class!r}"
        classes = classes[classes != exclude_class]
    if allocation == "asset_class":
        n_classes = classes.nunique()
        return 1.0 / (n_classes * classes.map(classes.value_counts()))
    if allocation == "instrument":
        return pd.Series(1.0 / len(classes), index=classes.index)
    raise ValueError(f"unknown allocation {allocation!r}")


def inverse_vol_weights(signal: pd.DataFrame, vol: pd.DataFrame, dates: pd.DatetimeIndex,
                        allocation: str, exclude_class: str | None = None,
                        classes: dict | None = None) -> pd.DataFrame:
    """Variant 2a: w_i = s_i * b_i * target / sigma_i.

    Each instrument's standalone ex-ante volatility is b_i x 10%, so the
    standalone vols sum to the 10% target. Realised portfolio volatility is
    lower because instruments are not perfectly correlated; 2a fixes the risk
    of each position, not of the portfolio.
    """
    b = risk_budgets(allocation, exclude_class, classes)
    s = signal.loc[dates, b.index]
    sig = vol.loc[dates, b.index]
    return s * b * PORTFOLIO_VOL_TARGET / sig


def portfolio_ex_ante_vol(weights: pd.DataFrame, daily_excess: pd.DataFrame,
                          vol: pd.DataFrame) -> pd.Series:
    """sqrt(w' Sigma w) at each date, Sigma from EWMA vols and trailing correlation."""
    out = {}
    for date, w in weights.iterrows():
        corr = trailing_correlation(daily_excess, date)
        cov = covariance(vol.loc[date, w.index], corr)
        out[date] = float(np.sqrt(w.values @ cov.values @ w.values))
    return pd.Series(out, name="ex_ante_vol")


def target_portfolio_vol(weights: pd.DataFrame, daily_excess: pd.DataFrame,
                         vol: pd.DataFrame, target: float = PORTFOLIO_VOL_TARGET):
    """Variant 2 (2b): rescale weights so ex-ante portfolio volatility equals target.

    Returns the scaled weights and the scale factor applied at each date. A
    date with no position (all signals zero) keeps zero weights.
    """
    pre = portfolio_ex_ante_vol(weights, daily_excess, vol)
    scale = (target / pre).where(pre > 0, 0.0)
    assert np.isfinite(scale).all(), "non-finite vol scaling"
    return weights.mul(scale, axis=0), scale


def risk_contributions(weights: pd.DataFrame, daily_excess: pd.DataFrame,
                       vol: pd.DataFrame) -> pd.DataFrame:
    """Ex-ante share of portfolio variance from each instrument, w_i (Sigma w)_i / w'Sigma w."""
    rows = {}
    for date, w in weights.iterrows():
        cov = covariance(vol.loc[date, w.index], trailing_correlation(daily_excess, date)).values
        marginal = cov @ w.values
        total = w.values @ marginal
        rows[date] = w.values * marginal / total if total > 0 else np.zeros(len(w))
    return pd.DataFrame(rows, index=weights.columns).T


def cap_gross_leverage(weights: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Scale down, pro rata, any date whose gross exposure sum |w_i| exceeds cap."""
    gross = weights.abs().sum(axis=1)
    factor = (cap / gross).clip(upper=1.0).where(gross > 0, 1.0)
    return weights.mul(factor, axis=0)
