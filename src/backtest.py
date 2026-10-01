"""Daily backtest engine with periodic rebalancing, weight drift and costs.

Accounting. The portfolio holds cash plus positions with notional w_i * NAV
(w_i can be negative or sum to more than one in absolute value). Cash earns
the T-bill rate and positions are financed at it, so the portfolio's return
in excess of cash on day d is

    R_d = sum_i w_{i,d-1} (r_{i,d} - rf_d)  -  costs_d  -  borrow_d,

where w_{d-1} are the weights at the close of day d-1, after any trade at
that close. Between trades the weights drift with prices:

    w_{i,d} = w_{i,d-1} (1 + r_{i,d}) / (1 + rf_d + R_d).

Trading. Target weights decided at the close of a signal date s are traded
at the close of the trading day `lag` days later (lag 0 = the same close).
The traded notional is |w_target - w_drifted| summed over instruments, and
costs are cost_bps per unit of notional traded, deducted on the trade day.
The first trade (building the book from zero) is charged like any other, on
the first day with P&L.
Short positions pay an annual borrow fee on their notional, accrued on an
actual/360 basis like the risk-free rate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    excess_returns: pd.Series      # daily R_d (net of costs and borrow fees)
    weights: pd.DataFrame          # end-of-day weights after any trade
    turnover: pd.Series            # traded notional on each trade date
    costs: pd.Series               # daily cost deduction (trading + borrow)
    rf: pd.Series                  # daily risk-free return used


def run_backtest(target_weights: pd.DataFrame, daily_total_returns: pd.DataFrame,
                 rf: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                 cost_bps: float = 0.0, borrow_bps: float = 0.0,
                 lag: int = 0) -> BacktestResult:
    """Simulate the portfolio from the first trade through `end`.

    target_weights: rows indexed by signal dates, columns = instruments.
    start: first signal date to trade. Returns are recorded from the day after
    the first trade through `end` inclusive.
    """
    days = daily_total_returns.loc[:end].index
    cols = list(target_weights.columns)
    signal_dates = target_weights.loc[start:end].index
    assert len(signal_dates) > 0, "no signal dates in range"
    assert set(signal_dates) <= set(days), "signal dates must be trading days"

    # Map each signal date to its execution day (lag trading days later).
    pos = days.get_indexer(signal_dates)
    exec_pos = pos + lag
    keep = exec_pos < len(days)
    exec_days = days[exec_pos[keep]]
    targets = target_weights.loc[signal_dates[keep], cols].values
    trades = dict(zip(exec_days, targets))

    first = exec_days[0]
    sim_days = days[days >= first]
    r = daily_total_returns.loc[sim_days, cols].values
    rf_v = rf.reindex(sim_days).values
    cal_days = sim_days.to_series().diff().dt.days.values

    n = len(sim_days)
    w = np.zeros(len(cols))
    out_ret = np.full(n, np.nan)
    out_w = np.zeros((n, len(cols)))
    out_cost = np.zeros(n)
    turnover = {}
    c = cost_bps / 1e4
    b = borrow_bps / 1e4

    pending_cost = 0.0
    for d in range(n):
        if d > 0:
            held = w != 0
            if np.isnan(r[d][held]).any() or np.isnan(rf_v[d]):
                raise ValueError(f"missing return for a held position on {sim_days[d].date()}")
            r_d = np.where(held, r[d], 0.0)
            borrow = b * np.sum(np.clip(-w, 0, None)) * cal_days[d] / 360.0
            gross = np.dot(w, r_d - rf_v[d])
            # The cost of building the book at the first close is charged on
            # the first P&L day, so it falls inside the reported sample.
            out_ret[d] = gross - borrow - pending_cost
            out_cost[d] = borrow + pending_cost
            w = w * (1 + r_d) / (1 + rf_v[d] + out_ret[d])
            pending_cost = 0.0
        if sim_days[d] in trades:
            target = np.nan_to_num(trades[sim_days[d]], nan=0.0)
            traded = np.sum(np.abs(target - w))
            turnover[sim_days[d]] = traded
            if d > 0:
                out_ret[d] -= c * traded
                out_cost[d] += c * traded
            else:
                pending_cost = c * traded
            w = target
        out_w[d] = w

    # Day 0 is the first trade's close: no P&L yet.
    sim_days, out_ret, out_w, out_cost, rf_v = (sim_days[1:], out_ret[1:], out_w[1:],
                                                out_cost[1:], rf_v[1:])

    idx = sim_days
    return BacktestResult(
        excess_returns=pd.Series(out_ret, idx, name="excess_return"),
        weights=pd.DataFrame(out_w, idx, cols),
        turnover=pd.Series(turnover, name="turnover"),
        costs=pd.Series(out_cost, idx, name="costs"),
        rf=pd.Series(rf_v, idx, name="rf"),
    )


def monthly_excess(daily_excess: pd.Series, rf: pd.Series) -> pd.Series:
    """Compound to monthly: prod(1 + rf + R) - prod(1 + rf) per calendar month.

    This is the monthly total return of the portfolio minus the monthly
    return on cash, the same definition used for the ETFs themselves.
    """
    df = pd.DataFrame({"R": daily_excess, "rf": rf.reindex(daily_excess.index).fillna(0.0)}).dropna()
    by_month = df.index.to_period("M")
    total = (1 + df["rf"] + df["R"]).groupby(by_month).prod()
    cash = (1 + df["rf"]).groupby(by_month).prod()
    out = total - cash
    out.index = df["R"].groupby(by_month).apply(lambda s: s.index[-1]).values
    return out.rename(daily_excess.name)
