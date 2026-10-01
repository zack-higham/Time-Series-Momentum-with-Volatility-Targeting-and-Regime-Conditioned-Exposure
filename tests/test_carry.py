"""Carry: values by hand, rank signal properties, and no lookahead."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carry import carry_panel, monthly_rates, rank_signal  # noqa: E402
from config import DXY_WEIGHTS, UNIVERSE_U2  # noqa: E402
from data_io import load_frame  # noqa: E402
from returns import load_prices_u2  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    prices = load_prices_u2()
    close = pd.concat([load_frame("close"), load_frame("u2add_close")], axis=1)[list(UNIVERSE_U2)]
    actions = pd.concat([load_frame("actions"), load_frame("u2add_actions")]).sort_index()
    carry = carry_panel(UNIVERSE_U2, close, actions, prices.index)
    return prices, close, actions, carry


def test_income_carry_by_hand(inputs):
    """SPY at 2019-12-31: distributions with ex-dates in (2018-12-31, 2019-12-31]
    over the close, minus the T-bill yield that day."""
    _, close, actions, carry = inputs
    t = pd.Timestamp("2019-12-31")
    d = actions[(actions["ticker"] == "SPY") & (actions["dividend"] > 0)]
    d = d.loc[(d.index > pd.Timestamp("2018-12-31")) & (d.index <= t), "dividend"]
    assert len(d) == 4                                   # quarterly payer
    irx = load_frame("irx")["irx_pct"].loc[:t].iloc[-1] / 100
    assert carry.at[t, "SPY"] == pytest.approx(d.sum() / close.at[t, "SPY"] - irx)


def test_fx_carry_uses_previous_month_rates(inputs):
    """FXA at 2010-06-30 = AUD minus USD 3-month rate for MAY 2010."""
    carry = inputs[3]
    r = load_frame("fred_monthly_rates")
    may = pd.Timestamp("2010-05-01")
    expected = (r.at[may, "IR3TIB01AUM156N"] - r.at[may, "IR3TIB01USM156N"]) / 100
    assert carry.at[pd.Timestamp("2010-06-30"), "FXA"] == pytest.approx(expected)


def test_uup_carry_is_us_minus_dxy_basket(inputs):
    carry = inputs[3]
    t = pd.Timestamp("2015-09-30")
    r = monthly_rates().loc[pd.Timestamp("2015-08-01")] / 100
    expected = r["USD"] - sum(w * r[c] for c, w in DXY_WEIGHTS.items())
    assert carry.at[t, "UUP"] == pytest.approx(expected)
    assert sum(DXY_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-3)


def test_sterling_splice_and_gap_fill():
    r = monthly_rates()
    sonia = load_frame("fred_daily")["IUDSOIA"]
    assert r.at[pd.Timestamp("2026-03-01"), "GBP"] == pytest.approx(sonia.loc["2026-03"].mean())
    raw = load_frame("fred_monthly_rates")["IR3TIB01USM156N"]
    assert np.isnan(raw.at[pd.Timestamp("2020-04-01")])
    assert r.at[pd.Timestamp("2020-04-01"), "USD"] == raw.at[pd.Timestamp("2020-03-01")]


def test_rank_signal_properties(inputs):
    carry = inputs[3]
    s = rank_signal(carry, UNIVERSE_U2).loc["2008-03-31":"2026-08-31"]
    assert s.abs().max().max() <= 1 + 1e-12
    for cls in ("Equity", "Rates", "FX"):
        cols = [t for t in s.columns if UNIVERSE_U2[t] == cls]
        assert np.allclose(s[cols].sum(axis=1), 0.0)          # rank-neutral within class
        top = carry[cols].loc["2015-12-31"].idxmax()
        assert s.at[pd.Timestamp("2015-12-31"), top] == pytest.approx(1.0)


def test_carry_has_no_lookahead(inputs):
    """Carry at dates before a cut is unchanged when every price, distribution
    and rate observation after the cut is deleted."""
    prices, close, actions, carry = inputs
    cut = pd.Timestamp("2020-03-18")
    short = carry_panel(UNIVERSE_U2, close.loc[:cut], actions.loc[:cut], prices.loc[:cut].index)
    dates = short.index[short.index < pd.Timestamp("2020-03-01")]
    pd.testing.assert_frame_equal(short.loc[dates], carry.loc[dates])
