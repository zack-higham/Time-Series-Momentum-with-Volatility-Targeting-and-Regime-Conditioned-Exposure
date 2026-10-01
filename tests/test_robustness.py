"""Stage 6 helpers: leverage cap, leave-one-class-out budgets, continuous
signal (by hand and no lookahead), crisis windows, deflated Sharpe ratio."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import deflated_sharpe_ratio  # noqa: E402
from metrics import drawdown_episode, period_excess  # noqa: E402
from portfolio import cap_gross_leverage, rebalance_dates, risk_budgets  # noqa: E402
from returns import daily_excess_returns, load_prices  # noqa: E402
from run_stage6 import drawdown_windows  # noqa: E402
from signals import continuous_tstat_signal  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    prices = load_prices()
    return prices, daily_excess_returns(prices), rebalance_dates(prices.index)


def test_leverage_cap_scales_only_dates_above_cap():
    w = pd.DataFrame([[1.0, -1.5], [0.5, 0.5]], columns=["A", "B"])
    out = cap_gross_leverage(w, 2.0)
    assert np.allclose(out.iloc[0], [0.8, -1.2])        # 2.5 gross -> 2.0, signs kept
    assert np.allclose(out.iloc[1], [0.5, 0.5])         # under the cap: unchanged


def test_leave_one_class_out_budgets():
    b = risk_budgets("asset_class", exclude_class="FX")
    assert set(b.index) == {"SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC"}
    assert b.sum() == pytest.approx(1.0)
    assert b["IEF"] == pytest.approx(1 / 6) and b["SPY"] == pytest.approx(1 / 9)


def test_continuous_signal_by_hand(inputs):
    _, dex, dates = inputs
    t = pd.Timestamp("2019-12-31")
    sig = continuous_tstat_signal(dex, pd.DatetimeIndex([t]))
    x = dex.loc["2019-01-01":"2019-12-31", "TLT"]        # previous month-end 2018-12-31 excluded
    tstat = x.mean() / (x.std() / np.sqrt(len(x)))
    assert sig.loc[t, "TLT"] == pytest.approx(np.clip(tstat / 2, -1, 1))
    assert sig.abs().max().max() <= 1.0


def test_continuous_signal_has_no_lookahead(inputs):
    _, dex, dates = inputs
    d = dates[(dates >= "2019-06-01") & (dates <= "2020-02-28")]
    full = continuous_tstat_signal(dex, d)
    short = continuous_tstat_signal(dex.loc[:"2020-03-18"], d)
    pd.testing.assert_frame_equal(full, short)


def test_drawdown_windows_on_toy_series():
    idx = pd.bdate_range("2020-01-01", periods=8)
    level = pd.Series([100, 110, 90, 80, 115, 100, 112, 95], index=idx, dtype=float)
    win = drawdown_windows(level, 0.15)
    # 110 -> 80 (-27%), recovered at 115; then 115 -> 95 (-17%), unrecovered.
    assert list(win["peak"]) == [idx[1], idx[4]]
    assert list(win["trough"]) == [idx[3], idx[7]]
    assert win["spy_dd"].iloc[0] == pytest.approx(80 / 110 - 1)
    assert win["recovered"].iloc[0] == idx[4] and pd.isna(win["recovered"].iloc[1])


def test_drawdown_episode_dates():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.1, -0.2, -0.1, 0.5, -0.01, 0.0], index=idx)
    ep = drawdown_episode(r)
    assert ep["mdd_peak"] == idx[0].date() and ep["mdd_trough"] == idx[2].date()
    assert ep["mdd_recovery"] == idx[3].date()


def test_period_excess_matches_compounding():
    idx = pd.bdate_range("2020-01-01", periods=3)
    r, rf = pd.Series([0.01, -0.02, 0.03], idx), pd.Series(0.001, idx)
    q = period_excess(r, rf, "Q")
    assert q.iloc[0] == pytest.approx(np.prod(1 + rf + r) - np.prod(1 + rf))


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0.01, 0.03, 240))
    srs = rng.normal(0.0, 0.05, 30)
    one = deflated_sharpe_ratio(m, srs, n_trials=2)
    many = deflated_sharpe_ratio(m, srs, n_trials=30)
    assert many["dsr"] < one["dsr"] and many["sr0_ann"] > 0
