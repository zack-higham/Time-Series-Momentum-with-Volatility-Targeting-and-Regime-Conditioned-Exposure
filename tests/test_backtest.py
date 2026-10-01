"""Backtest engine: hand-computed toy cases and a full-pipeline no-lookahead test."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import monthly_excess, run_backtest  # noqa: E402
from portfolio import raw_tsmom_weights, rebalance_dates  # noqa: E402
from returns import daily_returns, load_prices  # noqa: E402
from riskfree import daily_rf  # noqa: E402
from signals import tsmom_signal  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=6)


def frame(rows, cols=("A", "B")):
    return pd.DataFrame(rows, index=DAYS[:len(rows)], columns=list(cols), dtype=float)


def zero_rf():
    return pd.Series(0.0, index=DAYS)


def test_single_long_position_earns_asset_return():
    rets = frame([[0, 0], [0.10, 0], [-0.05, 0], [0.02, 0]])
    w = frame([[1, 0]])
    res = run_backtest(w, rets, zero_rf(), DAYS[0], DAYS[3])
    assert np.allclose(res.excess_returns.values, [0.10, -0.05, 0.02])
    assert np.allclose(res.weights["A"], 1.0)        # fully invested stays at 1


def test_two_asset_drift_by_hand():
    """50/50 book; A +10% on day 1, so day-2 weights are 0.55/1.05 and 0.5/1.05."""
    rets = frame([[0, 0], [0.10, 0.0], [0.0, 0.10]])
    w = frame([[0.5, 0.5]])
    res = run_backtest(w, rets, zero_rf(), DAYS[0], DAYS[2])
    assert res.excess_returns.iloc[0] == pytest.approx(0.05)
    assert res.excess_returns.iloc[1] == pytest.approx(0.5 / 1.05 * 0.10)
    total = (1 + res.excess_returns).prod() - 1
    assert total == pytest.approx(0.5 * 0.10 + 0.5 * 0.10)   # buy-and-hold identity


def test_short_position_and_its_drift():
    """Short 1x, asset +10%: lose 10%; short notional grows to 1.1 on NAV 0.9."""
    rets = frame([[0, 0], [0.10, 0], [0.10, 0]])
    w = frame([[-1, 0]])
    res = run_backtest(w, rets, zero_rf(), DAYS[0], DAYS[2])
    assert res.excess_returns.iloc[0] == pytest.approx(-0.10)
    assert res.weights["A"].iloc[0] == pytest.approx(-1.1 / 0.9)
    assert res.excess_returns.iloc[1] == pytest.approx(-1.1 / 0.9 * 0.10)


def test_cash_earns_rf_so_flat_book_has_zero_excess():
    rets = frame([[0, 0], [0.01, 0.02], [0.03, -0.01]])
    rf = pd.Series(0.0002, index=DAYS)
    res = run_backtest(frame([[0, 0]]), rets, rf, DAYS[0], DAYS[2])
    assert np.allclose(res.excess_returns, 0.0)


def test_excess_over_rf_for_long_position():
    rets = frame([[0, 0], [0.01, 0]])
    rf = pd.Series(0.0002, index=DAYS)
    res = run_backtest(frame([[1, 0]]), rets, rf, DAYS[0], DAYS[1])
    assert res.excess_returns.iloc[0] == pytest.approx(0.01 - 0.0002)


def test_costs_charged_once_per_unit_traded():
    """Build 1x long (cost 10 bps, charged on the first P&L day), then flip to
    1x short on day 2 (traded notional 2 after drift ~ 2, cost ~ 20 bps)."""
    rets = frame([[0, 0], [0.0, 0], [0.0, 0], [0.0, 0]])
    w = pd.DataFrame([[1, 0], [-1, 0]], index=[DAYS[0], DAYS[2]], columns=["A", "B"], dtype=float)
    res = run_backtest(w, rets, zero_rf(), DAYS[0], DAYS[3], cost_bps=10)
    assert res.excess_returns.iloc[0] == pytest.approx(-0.001)
    # Day-2 trade: drifted weight is 1 / (1 - 0.001) after the cost reduced NAV.
    drifted = 1 / (1 - 0.001)
    assert res.turnover.iloc[1] == pytest.approx(1 + drifted)
    assert res.excess_returns.iloc[1] == pytest.approx(-0.001 * (1 + drifted))
    assert res.excess_returns.iloc[2] == pytest.approx(0.0)


def test_execution_lag_delays_position_by_one_day():
    rets = frame([[0, 0], [0.05, 0], [0.02, 0]])
    res = run_backtest(frame([[1, 0]]), rets, zero_rf(), DAYS[0], DAYS[2], lag=1)
    # Traded at the close of day 1, so day 1's +5% is missed; day 2's +2% earned.
    assert np.allclose(res.excess_returns.values, [0.02])


def test_borrow_fee_on_shorts():
    rets = frame([[0, 0], [0.0, 0]])
    res = run_backtest(frame([[-2, 0]]), rets, zero_rf(), DAYS[0], DAYS[1], borrow_bps=36)
    # 2x short, 36 bps/yr, 1 calendar day on actual/360: 2 * 0.0036 / 360.
    assert res.excess_returns.iloc[0] == pytest.approx(-2 * 0.0036 / 360)


def test_financing_spread_on_borrowed_cash_only():
    """Long 1.5 + long 0.7, short -0.4: borrowed cash = 2.2 - 1 = 1.2 (shorts
    not netted). A 1x long book borrows nothing."""
    rets = frame([[0, 0, 0], [0.0, 0, 0]], cols=("A", "B", "C"))
    res = run_backtest(frame([[1.5, 0.7, -0.4]], cols=("A", "B", "C")), rets, zero_rf(),
                       DAYS[0], DAYS[1], financing_bps=36)
    assert res.excess_returns.iloc[0] == pytest.approx(-1.2 * 0.0036 / 360)
    res1 = run_backtest(frame([[1.0, 0]]), frame([[0, 0], [0.0, 0]]), zero_rf(),
                        DAYS[0], DAYS[1], financing_bps=36)
    assert res1.excess_returns.iloc[0] == 0.0


def test_missing_return_on_held_position_raises():
    rets = frame([[0, 0], [np.nan, 0]])
    with pytest.raises(ValueError):
        run_backtest(frame([[1, 0]]), rets, zero_rf(), DAYS[0], DAYS[1])


def test_monthly_excess_compounds_correctly():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"])
    R = pd.Series([0.01, 0.02, 0.03], idx)
    rf = pd.Series([0.001, 0.001, 0.001], idx)
    m = monthly_excess(R, rf)
    assert m.iloc[0] == pytest.approx(1.011 * 1.021 - 1.001 ** 2)
    assert m.index[0] == pd.Timestamp("2020-01-31")


def test_full_pipeline_has_no_lookahead():
    """V1 weights and daily returns up to T are identical when data after T is removed."""
    prices = load_prices()
    all_dates = rebalance_dates(prices.index)
    full = run_backtest(raw_tsmom_weights(tsmom_signal(prices, 12), all_dates),
                        daily_returns(prices), daily_rf(prices.index),
                        all_dates[0], prices.index[-1], cost_bps=10)
    for cut in ["2014-09-17", "2021-02-26"]:         # one mid-month, one month-end
        p = prices.loc[:cut]
        dates = all_dates[all_dates <= pd.Timestamp(cut)]   # a month-end cut trades too
        short = run_backtest(raw_tsmom_weights(tsmom_signal(p, 12), dates),
                             daily_returns(p), daily_rf(p.index),
                             dates[0], pd.Timestamp(cut), cost_bps=10)
        pd.testing.assert_series_equal(full.excess_returns.loc[:cut], short.excess_returns)
        pd.testing.assert_frame_equal(full.weights.loc[:cut], short.weights)
