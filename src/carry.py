"""Carry: the return an instrument would earn if prices did not change.

Following Koijen, Moskowitz, Pedersen and Vrugt (2018), carry is measured at
each month-end t from information known at t, annualised, in excess of cash:

  Equity and bond ETFs: trailing 12-month distributions per share divided by
    the closing price at t, minus the T-bill yield (^IRX) at t. Only
    distributions with ex-dates on or before t count. Yahoo reports both the
    closing price and the distributions split-adjusted, so their ratio is
    consistent across splits.
  Currency funds (long the foreign currency against USD): foreign minus US
    3-month interbank rate. The OECD series are monthly averages, so at
    month-end t the value for month t-1 is used: it is published by then.
    A missing month (USD, April 2020) takes the previous month's value.
    Sterling from 2026-02 uses the monthly mean of SONIA (the 3-month series
    ends in January 2026).
  UUP (long USD against the DXY basket): US rate minus the DXY-weighted
    foreign rate.
  Commodities: no carry measure (futures curves are not freely available).

Signal (primary): cross-sectional rank within each carry asset class,
    s_i = (rank_i - (n+1)/2) / ((n-1)/2) in [-1, 1], highest carry most long.
Robustness: time-series carry, s_i = sign(carry_i).
"""

import numpy as np
import pandas as pd

from config import (CARRY_CLASSES, CURRENCY_OF_FUND, DXY_WEIGHTS, FX_RATE_SERIES,
                    GBP_SPLICE_SERIES, SAMPLE_END)
from data_io import load_frame
from returns import month_end_dates

GBP_SPLICE_FROM = "2026-02-01"


def distribution_yield(close: pd.DataFrame, actions: pd.DataFrame,
                       dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Trailing 12-month distributions / price at each date (annual, decimal)."""
    div = actions[actions["dividend"] > 0]
    out = {}
    for t in close.columns:
        d = div.loc[div["ticker"] == t, "dividend"]
        vals = {}
        for date in dates:
            window = d.loc[(d.index > date - pd.DateOffset(years=1)) & (d.index <= date)]
            vals[date] = window.sum() / close.at[date, t]
        out[t] = pd.Series(vals)
    return pd.DataFrame(out)


def monthly_rates() -> pd.DataFrame:
    """Short rates by currency (annual %, monthly averages indexed by month
    start), with the sterling splice and single missing months filled."""
    raw = load_frame("fred_monthly_rates")
    rates = pd.DataFrame({ccy: raw[sid] for ccy, sid in FX_RATE_SERIES.items()})
    sonia = load_frame("fred_daily")[GBP_SPLICE_SERIES].resample("MS").mean()
    splice = sonia.loc[GBP_SPLICE_FROM:]
    splice = splice[splice.index.isin(rates.index)]   # a partial final month is never needed
    rates.loc[splice.index, "GBP"] = splice
    return rates.ffill(limit=1)


def rate_known_at(rates: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rates for month t-1, aligned to each month-end t."""
    prev_month = (dates.to_period("M") - 1).to_timestamp()
    out = rates.reindex(prev_month)
    out.index = dates
    return out


def fx_carry(dates: pd.DatetimeIndex, funds) -> pd.DataFrame:
    r = rate_known_at(monthly_rates(), dates) / 100.0
    out = {}
    for f in funds:
        if f == "UUP":
            foreign = sum(w * r[c] for c, w in DXY_WEIGHTS.items())
            out[f] = r["USD"] - foreign
        else:
            out[f] = r[CURRENCY_OF_FUND[f]] - r["USD"]
    return pd.DataFrame(out)


def carry_panel(classes: dict, close: pd.DataFrame, actions: pd.DataFrame,
                trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Annualised carry at every complete month-end for the funds in the carry
    classes; commodity funds are left out (no carry)."""
    dates = month_end_dates(trading_days)
    dates = dates[dates <= pd.Timestamp(SAMPLE_END)]
    income = [t for t, c in classes.items() if c in ("Equity", "Rates")]
    fx = [t for t, c in classes.items() if c == "FX"]
    irx = load_frame("irx")["irx_pct"].reindex(trading_days).ffill().reindex(dates) / 100.0
    dy = distribution_yield(close[income], actions, dates)
    out = pd.concat([dy.sub(irx, axis=0), fx_carry(dates, fx)], axis=1)
    return out[[t for t in classes if classes[t] in CARRY_CLASSES]]


def rank_signal(carry: pd.DataFrame, classes: dict) -> pd.DataFrame:
    """Cross-sectional rank within each asset class, scaled to [-1, 1]."""
    out = []
    for cls in CARRY_CLASSES:
        cols = [t for t in carry.columns if classes[t] == cls]
        x = carry[cols]
        n = x.notna().sum(axis=1)
        ranks = x.rank(axis=1, method="average")
        out.append(ranks.sub((n + 1) / 2, axis=0).div((n - 1) / 2, axis=0))
    return pd.concat(out, axis=1)[carry.columns]


def sign_signal(carry: pd.DataFrame) -> pd.DataFrame:
    return np.sign(carry)
