"""Tests for returns, volatility and signals: formulas, timing and no lookahead."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import EWMA_COM_DAYS  # noqa: E402
from predictive_regression import decompose, scaled_returns  # noqa: E402
from returns import (daily_excess_returns, load_prices, month_end_dates,  # noqa: E402
                     monthly_excess_returns, monthly_rf)
from riskfree import daily_rf  # noqa: E402
from signals import all_signals, tsmom_signal  # noqa: E402
from volatility import ewma_vol  # noqa: E402


@pytest.fixture(scope="module")
def prices():
    return load_prices()


def test_ewma_matches_explicit_formula():
    """pandas ewm(com=60, bias=True) equals the MOP weighted sum written out by hand."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 400))
    d = EWMA_COM_DAYS / (1 + EWMA_COM_DAYS)
    w = (1 - d) * d ** np.arange(len(r))[::-1]
    w = w / w.sum()                              # normalised weights (adjust=True)
    mean = np.sum(w * r.values)
    var_hand = np.sum(w * (r.values - mean) ** 2)
    vol = ewma_vol(r.to_frame("x"), min_obs=1)["x"].iloc[-1]
    assert vol == pytest.approx(np.sqrt(252 * var_hand), rel=1e-10)


def test_vol_and_signals_have_no_lookahead(prices):
    """Estimates dated <= T are identical whether or not data after T exists.

    T is chosen mid-month so a partial final month is also exercised: the
    signal at the last complete month-end must not change.
    """
    for cut in ["2015-06-17", "2020-03-31", "2022-11-09"]:
        short = prices.loc[:cut]
        full_vol = ewma_vol(daily_excess_returns(prices)).loc[:cut]
        short_vol = ewma_vol(daily_excess_returns(short))
        pd.testing.assert_frame_equal(full_vol, short_vol)

        # Complete month-ends on or before the cut (for a mid-month cut, the
        # partial month is excluded). The short run sees data ending at the
        # cut itself, partial month included.
        complete = month_end_dates(prices.index)
        complete = complete[complete <= pd.Timestamp(cut)]
        for k in (1, 3, 12):
            full_sig = tsmom_signal(prices, k).loc[complete]
            short_sig = tsmom_signal(short, k).loc[complete]
            pd.testing.assert_frame_equal(full_sig, short_sig)


def test_signal_by_hand(prices):
    """12m signal for SPY at 2020-03-31 and TLT at 2022-06-30 recomputed from raw inputs."""
    sig = tsmom_signal(prices, 12)
    rf = daily_rf(prices.index).fillna(0)
    for ticker, end, start in [("SPY", "2020-03-31", "2019-03-29"), ("TLT", "2022-06-30", "2021-06-30")]:
        p = prices[ticker]
        total = p.loc[end] / p.loc[start] - 1
        cash = (1 + rf.loc[pd.Timestamp(start) + pd.Timedelta(days=1):end]).prod() - 1
        assert sig.loc[end, ticker] == np.sign(total - cash)
    # SPY fell ~7% over the year to March 2020; TLT fell ~18% to June 2022.
    assert sig.loc["2020-03-31", "SPY"] == -1
    assert sig.loc["2022-06-30", "TLT"] == -1


def test_monthly_excess_return_by_hand(prices):
    m = monthly_excess_returns(prices)
    rf = daily_rf(prices.index).fillna(0)
    p = prices["IEF"]
    total = p.loc["2023-07-31"] / p.loc["2023-06-30"] - 1
    cash = (1 + rf.loc["2023-07-01":"2023-07-31"]).prod() - 1
    assert m.loc["2023-07-31", "IEF"] == pytest.approx(total - cash, abs=1e-14)
    assert 0.0035 < cash < 0.0050                # ~5.2% a year in mid-2023


def test_monthly_rf_covers_each_day_once(prices):
    """Compounded monthly cash equals compounded daily cash over the whole sample."""
    days = prices.index
    rf_d = daily_rf(days).fillna(0)
    assert (1 + monthly_rf(days)).prod() == pytest.approx((1 + rf_d).prod(), rel=1e-12)


def test_scaled_return_uses_previous_month_vol(prices):
    ends = month_end_dates(prices.index)
    m = monthly_excess_returns(prices)
    vol = ewma_vol(daily_excess_returns(prices)).loc[ends]
    z = scaled_returns(m, vol)
    t, prev = pd.Timestamp("2020-03-31"), pd.Timestamp("2020-02-28")
    assert z.loc[t, "SPY"] == pytest.approx(m.loc[t, "SPY"] / vol.loc[prev, "SPY"])


def test_blend_requires_all_horizons(prices):
    s = all_signals(prices)
    blend = s["blend"]
    any_missing = pd.concat([s[k].isna() for k in (1, 3, 6, 12)]).groupby(level=0).any()
    assert blend[any_missing.reindex(blend.index)].isna().all().all()


def test_decomposition_adds_up():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    z = pd.DataFrame(rng.normal(0.02, 0.3, (200, 2)), index=idx, columns=["a", "b"])
    x = np.sign(pd.DataFrame(rng.normal(0.1, 1, (200, 2)), index=idx, columns=["a", "b"]))
    d = decompose(z, x, idx[1:])
    assert np.allclose(d["total"], d["timing"] + d["static"])
