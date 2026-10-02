"""Portfolio application: trend following as a diversifier.

Holds V2 alongside a conventional portfolio at a weight fixed in advance
(config.py, BLEND_*): 80% 60/40 + 20% V2 (primary) and 80% SPY + 20% V2
(secondary). Added after the main results had been seen and specified before
it was computed. V2 itself is unchanged, so this is not a further trend
configuration for the deflated Sharpe ratio.

Accounting follows the engine. Each sleeve earns its own daily excess return;
the sleeve weights are reset to the targets at the close of each rebalance
date and drift with the sleeves' total returns in between:

    R_d = a_{d-1} R_bench,d + (1 - a_{d-1}) R_V2,d
    a_d = a_{d-1} (1 + rf_d + R_bench,d) / (1 + rf_d + R_d)

Inputs are the saved Stage 6 daily excess returns (V2 net of 10 bps; 60/40
and SPY gross), so nothing upstream is recomputed.
"""

import numpy as np
import pandas as pd

from backtest import monthly_excess
from config import BLEND_TREND_SERIES, BLEND_WEIGHT_TREND, BLENDS, OUTPUT_DIR, SAMPLE_END
from data_io import load_frame, save_output
from inference import sharpe_difference_test
from metrics import drawdown_episode, lo_sharpe_se, period_excess, summary
from portfolio import rebalance_dates
from returns import load_prices
from riskfree import daily_rf
from run_stage6 import window_return

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def blend_daily(bench: pd.Series, trend: pd.Series, rf: pd.Series, rebalance: pd.DatetimeIndex,
                w_trend: float) -> tuple[pd.Series, pd.DataFrame]:
    """Daily blend excess returns and the sleeve trades at each rebalance."""
    a_target = 1.0 - w_trend
    days = bench.index
    assert days.equals(trend.index)
    assert rebalance[0] < days[0], "the book is built at the close before the first return day"
    a = a_target
    out = np.empty(len(days))
    trades = []
    rb = set(rebalance)
    for i, d in enumerate(days):
        r = a * bench.iloc[i] + (1 - a) * trend.iloc[i]
        out[i] = r
        a = a * (1 + rf.loc[d] + bench.iloc[i]) / (1 + rf.loc[d] + r)
        if d in rb:
            trades.append({"date": d, "drifted_bench_weight": a, "sleeve_turnover": 2 * abs(a - a_target)})
            a = a_target
    return pd.Series(out, days), pd.DataFrame(trades).set_index("date")


def main():
    prices = load_prices()
    rf = daily_rf(prices.index)
    dates = rebalance_dates(prices.index)
    daily = load_frame("stage6_daily_returns", OUTPUT_DIR)
    monthly = load_frame("stage6_monthly_returns", OUTPUT_DIR)
    windows = load_frame("stage6_crisis_windows", OUTPUT_DIR)
    leverage = load_frame("stage4_leverage", OUTPUT_DIR)["gross"]
    trend = BLEND_TREND_SERIES

    b_daily, b_monthly, turnover = {}, {}, []
    for name, bench in BLENDS.items():
        d, trades = blend_daily(daily[bench], daily[trend], rf, dates, BLEND_WEIGHT_TREND)
        m = monthly_excess(d.rename(name), rf)
        assert m.index.equals(monthly.index), "blend months must match the main sample"
        b_daily[name], b_monthly[name] = d, m
        # Sleeve trade (NAV moved between sleeves) and an upper bound on the
        # instrument notional it implies: each sleeve's trade times its gross
        # leverage (benchmarks are fully invested, gross 1; V2 at its target).
        moved = trades["sleeve_turnover"] / 2
        notional = moved * 1.0 + moved * leverage.reindex(trades.index)
        turnover.append({"blend": name, "rebalances": len(trades),
                         "sleeve_turnover_yr": trades["sleeve_turnover"].mean() * 12,
                         "notional_upper_bound_yr": notional.mean() * 12,
                         "cost_at_10bps_upper_bound_pct_yr": notional.mean() * 12 * 10 / 100,
                         "v2_turnover_yr_for_comparison": None})
    b_daily, b_monthly = pd.DataFrame(b_daily), pd.DataFrame(b_monthly)
    core = pd.read_csv(OUTPUT_DIR / "stage6_core_metrics.csv", index_col=0)
    for row in turnover:
        row["v2_turnover_yr_for_comparison"] = core.loc[trend, "turnover_yr"]
    turnover = pd.DataFrame(turnover).set_index("blend")

    # --- alignment checks
    m_all = pd.concat([monthly[["60/40", "SPY", trend]], b_monthly], axis=1)
    d_all = pd.concat([daily[["60/40", "SPY", trend]], b_daily], axis=1)
    assert len(m_all) == 222 and m_all.notna().all().all()
    assert m_all.index[0] == pd.Timestamp("2008-04-30") and m_all.index[-1] == pd.Timestamp(SAMPLE_END)
    print(f"Months: {len(m_all)}, {m_all.index[0].date()} to {m_all.index[-1].date()}; "
          f"daily obs {len(d_all)}, {d_all.index[0].date()} to {d_all.index[-1].date()}; "
          f"rebalances {turnover['rebalances'].iloc[0]}")

    # With month-end rebalancing and buy-and-hold sleeves within each month, blend
    # wealth over a month is exactly 0.8 x benchmark wealth + 0.2 x V2 wealth, so
    # monthly excess returns must be exactly linear in the sleeves' (drift does not
    # break this; it matters only inside the month, i.e. for daily drawdowns).
    for name, bench in BLENDS.items():
        approx = (1 - BLEND_WEIGHT_TREND) * monthly[bench] + BLEND_WEIGHT_TREND * monthly[trend]
        diff = (b_monthly[name] - approx).abs()
        print(f"{name}: max |blend - 0.8 bench - 0.2 V2| = {diff.max() * 100:.3f}%/month "
              f"(mean {diff.mean() * 100:.4f}%)")

    order = ["60/40", "60/40+V2", "SPY", "SPY+V2"]
    rows = []
    for n in order:
        st = summary(m_all[n], d_all[n])
        st["sharpe_se_lo"] = lo_sharpe_se(m_all[n])
        st.update(drawdown_episode(d_all[n]))
        rows.append({"portfolio": n, **st})
    metrics = pd.DataFrame(rows).set_index("portfolio")
    for name, bench in BLENDS.items():
        metrics.loc[bench, "corr_v2"] = m_all[trend].corr(m_all[bench])

    # First-principles volatility check
    for name, bench in BLENDS.items():
        sb, sv = m_all[bench].std(), m_all[trend].std()
        rho = m_all[trend].corr(m_all[bench])
        w = BLEND_WEIGHT_TREND
        pred = np.sqrt((1 - w) ** 2 * sb ** 2 + w ** 2 * sv ** 2 + 2 * w * (1 - w) * rho * sb * sv) * np.sqrt(12)
        print(f"{name}: vol from weights and correlation {pred * 100:.2f}% vs realised "
              f"{metrics.loc[name, 'ann_vol_pct']:.2f}% (0.8 x benchmark alone: "
              f"{0.8 * metrics.loc[bench, 'ann_vol_pct']:.2f}%; rho {rho:.3f})")

    # Sharpe difference tests (same bootstrap as Table 5)
    tests = pd.DataFrame([{"blend": name, "benchmark": bench, **sharpe_difference_test(m_all[name], m_all[bench])}
                          for name, bench in BLENDS.items()]).set_index("blend")

    # Crisis windows (Stage 6 definitions) and SPY worst-decile periods
    crisis = windows[["trough", "spy_dd"]].copy()
    for n in order:
        crisis[n] = [window_return(d_all[n], p, pd.Timestamp(t)) * 100 for p, t in crisis["trough"].items()]
        assert n not in ("60/40", "SPY") or np.allclose(crisis[n], windows[n]), "crisis windows must match Stage 6"
    q_all = pd.DataFrame({n: period_excess(d_all[n], rf, "Q") for n in order})
    assert len(q_all) == 74
    cond = []
    for label, frame in (("month", m_all), ("quarter", q_all)):
        worst = frame["SPY"] <= frame["SPY"].quantile(0.10)
        for n in order:
            cond.append({"freq": label, "portfolio": n, "n_worst": int(worst.sum()),
                         "mean_worst_pct": frame.loc[worst, n].mean() * 100,
                         "mean_rest_pct": frame.loc[~worst, n].mean() * 100})
    cond = pd.DataFrame(cond).set_index(["freq", "portfolio"])
    ref = pd.read_csv(OUTPUT_DIR / "stage6_worst_decile.csv", index_col=[0, 1])
    for f in ("month", "quarter"):
        for n in ("60/40", "SPY"):
            assert np.isclose(cond.loc[(f, n), "mean_worst_pct"], ref.loc[(f, n), "mean_worst_pct"])

    save_output(b_daily, "blend_daily_returns")
    save_output(b_monthly, "blend_monthly_returns")
    save_output(metrics, "blend_metrics")
    save_output(tests, "blend_sharpe_tests")
    save_output(crisis, "blend_crisis_windows")
    save_output(cond, "blend_worst_decile")
    save_output(turnover, "blend_turnover")

    print("\nPerformance (excess of T-bills; V2 net of 10 bps, benchmarks gross):")
    print(metrics[["ann_ret_pct", "ann_vol_pct", "sharpe", "sharpe_se_lo", "max_dd_pct", "calmar",
                   "skew_m", "mdd_peak", "mdd_trough", "mdd_recovery", "corr_v2"]].round(3).to_string())
    print("\nSharpe difference, blend minus benchmark (paired stationary bootstrap):")
    print(tests.round(3).to_string())
    print("\nCrisis windows, cumulative excess return %:")
    print(crisis.round(2).to_string())
    print("\nSPY worst-decile months and quarters, mean excess return %:")
    print(cond.round(2).to_string())
    print("\nBlend rebalancing turnover:")
    print(turnover.round(3).to_string())
    for name, bench in BLENDS.items():
        better = [(pd.Timestamp(p).year, crisis.loc[p, name] > crisis.loc[p, bench]) for p in crisis.index]
        print(f"{name}: smaller crisis loss than {bench} in windows: {better}")


if __name__ == "__main__":
    main()
