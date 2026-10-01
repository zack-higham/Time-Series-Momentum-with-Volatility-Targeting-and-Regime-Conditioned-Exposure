"""Regime model: filtered probabilities use no future data; labels are identified."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import (daily_trend_payoff, fit_markov, target_state, walk_forward,  # noqa: E402
                    weekly_trend_payoff)
from returns import load_prices  # noqa: E402


def synthetic(seed=0, n=600):
    """Two persistent regimes: calm/low mean, then volatile/high mean, alternating."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = state[i - 1] if rng.random() < 0.97 else 1 - state[i - 1]
    mu, sd = np.array([-0.002, 0.004]), np.array([0.005, 0.015])
    return rng.normal(mu[state], sd[state]), state


def test_filtered_probability_ignores_later_observations():
    """With parameters held fixed, P(S_t | y_1..y_t) is identical whether the
    series stops at t or continues; smoothed probabilities are not."""
    y, _ = synthetic()
    model, res, _ = fit_markov(y)
    longer = model.__class__(y * 100, k_regimes=2, trend="c", switching_variance=True)
    shorter = model.__class__(y[:400] * 100, k_regimes=2, trend="c", switching_variance=True)
    f_long = np.asarray(longer.filter(res.params).filtered_marginal_probabilities)
    f_short = np.asarray(shorter.filter(res.params).filtered_marginal_probabilities)
    assert np.allclose(f_long[:400], f_short)
    # Smoothed probabilities condition on the whole sample, so around regime
    # changes they differ materially from the filtered ones: they look ahead.
    s_long = np.asarray(longer.smooth(res.params).smoothed_marginal_probabilities)
    assert np.max(np.abs(s_long[:, 0] - f_long[:, 0])) > 0.2


def test_state_labels_identified_on_synthetic_data():
    y, state = synthetic(seed=3)
    _, res, params = fit_markov(y)
    j = target_state(params, "mean")
    p = np.asarray(res.filtered_marginal_probabilities)[:, j]
    accuracy = np.mean((p > 0.5) == (state == 1))
    assert params[f"const[{j}]"] > params[f"const[{1 - j}]"]
    assert accuracy > 0.85
    assert target_state(params, "variance") == j    # high-mean state is also high-variance here


@pytest.fixture(scope="module")
def prices():
    return load_prices()


def test_trend_payoff_uses_previous_month_signal(prices):
    """On the first trading day of a month the payoff uses the signal set at the
    previous month-end; on the month-end itself it still uses the older one."""
    from returns import daily_excess_returns
    from signals import tsmom_signal
    from volatility import ewma_vol
    d = daily_trend_payoff(prices)
    dex = daily_excess_returns(prices)
    vol = ewma_vol(dex)
    sig = tsmom_signal(prices, 12)
    for day, set_on in [("2020-04-01", "2020-03-31"), ("2020-03-31", "2020-02-28")]:
        expected = (sig.loc[set_on] * dex.loc[day] / vol.loc[set_on]).mean()
        assert d.loc[day, "payoff"] == pytest.approx(expected)


def test_walk_forward_has_no_lookahead(prices):
    """Walk-forward probabilities at dates <= cut are identical when the price
    data after the cut is deleted."""
    cut = "2020-03-18"
    dates = pd.DatetimeIndex(["2012-06-29", "2019-12-31", "2020-02-28"])
    full = walk_forward(weekly_trend_payoff(prices), dates, "mean")
    short = walk_forward(weekly_trend_payoff(prices.loc[:cut]), dates, "mean")
    pd.testing.assert_series_equal(full["p"], short["p"])
