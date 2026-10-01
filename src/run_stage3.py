"""Stage 3: Variant 1 (raw TSMOM, equal notional) for every lookback.

Writes to output/:
  v1_daily_returns.csv, v1_monthly_returns.csv   net of 10 bps costs
  v1_gross_monthly_returns.csv                   no costs
  v1_turnover.csv                                traded notional per rebalance
  v1_summary.csv                                 statistics per lookback
"""

import numpy as np
import pandas as pd

from backtest import monthly_excess, run_backtest
from config import COST_BPS, LOOKBACKS_MONTHS, MAIN_START, SAMPLE_END
from data_io import save_output
from metrics import summary
from portfolio import raw_tsmom_weights, rebalance_dates
from returns import daily_returns, load_prices, monthly_excess_returns
from riskfree import daily_rf
from signals import all_signals

HORIZONS = list(LOOKBACKS_MONTHS) + ["blend"]


def main():
    prices = load_prices()
    rets = daily_returns(prices)
    rf = daily_rf(prices.index)
    signals = all_signals(prices)
    dates = rebalance_dates(prices.index)
    end = pd.Timestamp(SAMPLE_END)
    print(f"Rebalances: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")

    daily, monthly, gross_monthly, turnover, rows = {}, {}, {}, {}, []
    for k in HORIZONS:
        w = raw_tsmom_weights(signals[k], dates)
        assert w.notna().all().all(), f"missing signal in main sample for lookback {k}"
        net = run_backtest(w, rets, rf, dates[0], end, cost_bps=COST_BPS)
        gross = run_backtest(w, rets, rf, dates[0], end, cost_bps=0.0)
        name = f"k{k}"
        daily[name] = net.excess_returns
        monthly[name] = monthly_excess(net.excess_returns, net.rf)
        gross_monthly[name] = monthly_excess(gross.excess_returns, gross.rf)
        turnover[name] = net.turnover
        assert monthly[name].index[0].strftime("%Y-%m") == MAIN_START
        assert len(monthly[name]) == 222, "expected 222 complete months"
        stats = summary(monthly[name], net.excess_returns)
        stats["gross_sharpe"] = (gross_monthly[name].mean() / gross_monthly[name].std()
                                 * np.sqrt(12))
        # Exclude the initial build from the ongoing turnover rate.
        stats["turnover_yr"] = net.turnover.iloc[1:].mean() * 12
        stats["mean_gross_exposure"] = net.weights.abs().sum(axis=1).mean()
        rows.append({"lookback": name, **stats})

    table = pd.DataFrame(rows).set_index("lookback")
    save_output(pd.DataFrame(daily), "v1_daily_returns")
    save_output(pd.DataFrame(monthly), "v1_monthly_returns")
    save_output(pd.DataFrame(gross_monthly), "v1_gross_monthly_returns")
    save_output(pd.DataFrame(turnover), "v1_turnover")
    save_output(table, "v1_summary")
    pd.set_option("display.width", 200)
    print("\nVariant 1 (raw TSMOM, equal notional, net of 10 bps):")
    print(table.round(3).to_string())

    # Cross-check against a simple monthly calculation that ignores intra-month
    # drift and costs: (1/N) sum_i s_{i,t-1} r_{i,t}.
    m_ex = monthly_excess_returns(prices).loc[MAIN_START:SAMPLE_END]
    simple = (signals[12].shift(1).loc[m_ex.index] * m_ex).mean(axis=1)
    g = gross_monthly["k12"]
    diff = (g - simple)
    print(f"\nEngine (gross, 12m) vs simple monthly product: corr {g.corr(simple):.5f}, "
          f"mean diff {diff.mean() * 1e4:+.2f} bps/month, max abs diff {diff.abs().max() * 1e4:.1f} bps")


if __name__ == "__main__":
    main()
