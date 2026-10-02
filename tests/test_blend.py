"""Blend accounting: sleeve drift within the month, reset at rebalance closes."""

import numpy as np
import pandas as pd

from run_blend import blend_daily


def toy():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, len(days)), days)
    trend = pd.Series(rng.normal(0, 0.01, len(days)), days)
    rf = pd.Series(0.0001, days)
    ends = days.to_series().groupby(days.to_period("M")).max()
    rebalance = pd.DatetimeIndex([pd.Timestamp("2019-12-31")]).append(pd.DatetimeIndex(ends.values[:-1]))
    return bench, trend, rf, rebalance


def test_monthly_wealth_is_exactly_the_weighted_sleeves():
    bench, trend, rf, rebalance = toy()
    d, _ = blend_daily(bench, trend, rf, rebalance, 0.2)
    for _, idx in d.groupby(d.index.to_period("M")).groups.items():
        g = lambda x: (1 + rf.loc[idx] + x.loc[idx]).prod()
        assert np.isclose(g(d), 0.8 * g(bench) + 0.2 * g(trend))


def test_weights_drift_then_reset():
    bench, trend, rf, rebalance = toy()
    _, trades = blend_daily(bench, trend, rf, rebalance, 0.2)
    assert len(trades) == 2                       # January and February month-ends
    assert (trades["drifted_bench_weight"] != 0.8).all()
    assert np.allclose(trades["sleeve_turnover"], 2 * (trades["drifted_bench_weight"] - 0.8).abs())


def test_first_day_uses_target_weights():
    bench, trend, rf, rebalance = toy()
    d, _ = blend_daily(bench, trend, rf, rebalance, 0.2)
    assert np.isclose(d.iloc[0], 0.8 * bench.iloc[0] + 0.2 * trend.iloc[0])
