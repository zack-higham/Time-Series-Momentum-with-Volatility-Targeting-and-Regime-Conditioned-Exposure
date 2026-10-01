"""Volatility targeting: weights by hand, ex-ante target hit, no lookahead."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import sharpe_difference_test, stationary_bootstrap_indices  # noqa: E402
from portfolio import (inverse_vol_weights, portfolio_ex_ante_vol, rebalance_dates,  # noqa: E402
                       risk_budgets, risk_contributions, target_portfolio_vol)
from returns import daily_excess_returns, load_prices  # noqa: E402
from signals import tsmom_signal  # noqa: E402
from volatility import ewma_vol  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    prices = load_prices()
    dex = daily_excess_returns(prices)
    vol = ewma_vol(dex)
    sig = tsmom_signal(prices, 12)
    dates = rebalance_dates(prices.index)
    return prices, dex, vol, sig, dates


def test_risk_budgets():
    b = risk_budgets("asset_class")
    assert b.sum() == pytest.approx(1.0)
    assert b["SPY"] == pytest.approx(0.25 / 3) and b["IEF"] == pytest.approx(0.25 / 2)
    assert risk_budgets("instrument").sum() == pytest.approx(1.0)


def test_inverse_vol_weight_by_hand(inputs):
    _, _, vol, sig, dates = inputs
    w = inverse_vol_weights(sig, vol, dates, "asset_class")
    d = pd.Timestamp("2015-06-30")
    expected = sig.loc[d, "TLT"] * 0.125 * 0.10 / vol.loc[d, "TLT"]
    assert w.loc[d, "TLT"] == pytest.approx(expected)


def test_scaled_portfolio_hits_target_with_independent_covariance(inputs):
    """Rebuild the covariance with numpy (not the project's functions)."""
    _, dex, vol, sig, dates = inputs
    w2, _ = target_portfolio_vol(inverse_vol_weights(sig, vol, dates, "asset_class"), dex, vol)
    for d in [pd.Timestamp("2008-10-31"), pd.Timestamp("2019-01-31"), pd.Timestamp("2026-08-31")]:
        window = dex.loc[:d].tail(252).values
        corr = np.corrcoef(window.T)
        sd = vol.loc[d, w2.columns].values
        cov = np.outer(sd, sd) * corr
        wv = w2.loc[d].values
        assert np.sqrt(wv @ cov @ wv) == pytest.approx(0.10, rel=1e-10)


def test_risk_contributions_sum_to_one(inputs):
    _, dex, vol, sig, dates = inputs
    w = inverse_vol_weights(sig, vol, dates[:5], "asset_class")
    rc = risk_contributions(w, dex, vol)
    assert np.allclose(rc.sum(axis=1), 1.0)


def test_vol_targeted_weights_have_no_lookahead(inputs):
    prices, _, _, _, dates = inputs
    cut = "2020-03-18"                                    # mid-crisis, mid-month
    short = prices.loc[:cut]
    results = []
    for p in (prices, short):
        dex = daily_excess_returns(p)
        vol = ewma_vol(dex)
        d = dates[dates <= pd.Timestamp(cut)]
        w2, _ = target_portfolio_vol(inverse_vol_weights(tsmom_signal(p, 12), vol, d, "asset_class"),
                                     dex, vol)
        results.append(w2)
    pd.testing.assert_frame_equal(results[0], results[1])


def test_stationary_bootstrap_block_structure():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(10_000, 6.0, rng)
    breaks = np.mean(np.diff(idx) % 10_000 != 1)
    assert breaks == pytest.approx(1 / 6, abs=0.01)       # mean block length ~6


def test_sharpe_difference_identical_series_is_zero():
    x = pd.Series(np.random.default_rng(2).normal(0.005, 0.03, 200))
    res = sharpe_difference_test(x, x, reps=200)
    assert res["diff"] == 0.0 and res["p_value"] == 1.0
