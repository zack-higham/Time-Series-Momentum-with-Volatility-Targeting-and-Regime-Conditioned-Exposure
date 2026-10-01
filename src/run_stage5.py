"""Stage 5: regime-conditioned exposure (Variant 3).

  V3      = V2 weights x (0.5 + P(trending)),   R1 walk-forward, burn-in from 2003
  V3_R2   = V2 weights x (0.5 + P(turbulent)),  R2 (SPY) walk-forward
  V3_main = as V3, HMM trained on main-sample weeks only (live from ~2011;
            multiplier 1 before that)
Diagnostic (added after the first run, not pre-registered):
  V3_nodegen = as V3, but multiplier 1 at refits whose fit is degenerate
               (a state with expected duration under two weeks)
Lookahead illustration (NOT results):
  V3_LA_smoothed   full-sample fit, smoothed probabilities
  V3_LA_fullparams full-sample parameters, filtered probabilities
Writes output/stage5_*.csv; report via tee to output/stage5_report.txt.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import monthly_excess, run_backtest
from config import COST_BPS, MAIN_START, REGIME_FLOOR, SAMPLE_END, newey_west_lags
from data_io import save_output
from inference import sharpe_difference_test, spanning_regression
from metrics import summary
from portfolio import inverse_vol_weights, rebalance_dates, target_portfolio_vol
from regime import (exposure_multiplier, full_sample_probabilities, walk_forward,
                    weekly_spy_excess, weekly_trend_payoff)
from returns import daily_excess_returns, daily_returns, load_prices
from riskfree import daily_rf
from signals import tsmom_signal
from volatility import ewma_vol

pd.set_option("display.width", 220)


def last_week_value(weekly_values: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Value of the last week ending on or before each date."""
    return pd.Series([weekly_values.loc[:t].iloc[-1] for t in dates], index=dates)


def predictive_test(next_month: pd.Series, p: pd.Series) -> dict:
    """V2 return in month t+1 regressed on p_t (NW t), and conditional means."""
    df = pd.DataFrame({"r": next_month.values, "p": p.values}).dropna()
    fit = sm.OLS(df["r"], sm.add_constant(df["p"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))})
    hi, lo = df[df.p > 0.5]["r"], df[df.p <= 0.5]["r"]
    return {"slope_pct": fit.params["p"] * 100, "t": fit.tvalues["p"],
            "mean_hi_pct": hi.mean() * 100, "sharpe_hi": hi.mean() / hi.std() * np.sqrt(12),
            "n_hi": len(hi), "mean_lo_pct": lo.mean() * 100,
            "sharpe_lo": lo.mean() / lo.std() * np.sqrt(12), "n_lo": len(lo)}


def main():
    prices = load_prices()
    rets, rf = daily_returns(prices), daily_rf(prices.index)
    dex = daily_excess_returns(prices)
    vol = ewma_vol(dex)
    dates = rebalance_dates(prices.index)
    end = pd.Timestamp(SAMPLE_END)
    w2, _ = target_portfolio_vol(
        inverse_vol_weights(tsmom_signal(prices, 12), vol, dates, "asset_class"), dex, vol)

    # --- Observation series ---------------------------------------------------
    tp = weekly_trend_payoff(prices).loc[:SAMPLE_END]
    spy = weekly_spy_excess(prices).loc[:SAMPLE_END]
    print(f"Trend payoff: {len(tp)} weeks {tp.index[0].date()} to {tp.index[-1].date()}; "
          f"instruments at start {int(tp.n_instruments.iloc[0])}, "
          f"all 10 from {tp.index[tp.n_instruments == 10][0].date()}")
    burn = tp.loc[:dates[0]]
    print(f"Burn-in weeks before first rebalance: {len(burn)}; "
          f"by instrument count: {burn.n_instruments.value_counts().sort_index().to_dict()}")
    save_output(tp, "stage5_weekly_trend_payoff")

    # --- Walk-forward fits ----------------------------------------------------
    wf = {"R1": walk_forward(tp, dates, "mean"),
          "R2": walk_forward(spy, dates, "variance"),
          "R1_main": walk_forward(tp.loc[f"{MAIN_START}-01":], dates, "mean")}
    for name, df in wf.items():
        save_output(df, f"stage5_walkforward_{name}")
    assert wf["R1"]["p"].notna().all() and wf["R2"]["p"].notna().all()
    for name, df in wf.items():
        assert (df["last_week"].dropna() <= df.index[df["last_week"].notna()]).all()
    live_main = wf["R1_main"]["p"].first_valid_index()
    print(f"R1_main live from {live_main.date()}")
    for name, df in wf.items():
        deg = df.index[df["degenerate"].eq(True)]
        span_txt = f" ({deg[0].date()} to {deg[-1].date()})" if len(deg) else ""
        print(f"{name}: degenerate refits {len(deg)} of {df['p'].notna().sum()}{span_txt}")
    p_nodegen = wf["R1"]["p"].where(wf["R1"]["degenerate"].eq(False))   # NaN -> multiplier 1

    # --- Lookahead illustration ---------------------------------------------
    la = full_sample_probabilities(tp, "mean")
    p_smooth = last_week_value(la["smoothed"], dates)
    p_fullf = last_week_value(la["filtered_fullparams"], dates)

    probs = pd.DataFrame({"R1": wf["R1"]["p"], "R2": wf["R2"]["p"],
                          "R1_main": wf["R1_main"]["p"], "R1_nodegen": p_nodegen,
                          "LA_smoothed": p_smooth,
                          "LA_fullparams": p_fullf})
    save_output(probs, "stage5_probabilities")

    # --- Overlays -------------------------------------------------------------
    weights = {"V2": w2}
    for name, col in [("V3", "R1"), ("V3_R2", "R2"), ("V3_main", "R1_main"),
                      ("V3_nodegen", "R1_nodegen"),
                      ("V3_LA_smoothed", "LA_smoothed"), ("V3_LA_fullparams", "LA_fullparams")]:
        m = exposure_multiplier(probs[col], REGIME_FLOOR).fillna(1.0)   # 1 = no overlay yet
        weights[name] = w2.mul(m, axis=0)
    monthly, rows = {}, []
    for name, wt in weights.items():
        res = run_backtest(wt, rets, rf, dates[0], end, cost_bps=COST_BPS)
        monthly[name] = monthly_excess(res.excess_returns, res.rf)
        st = summary(monthly[name], res.excess_returns)
        st["turnover_yr"] = res.turnover.iloc[1:].mean() * 12
        st["mean_gross_lev"] = wt.abs().sum(axis=1).mean()
        rows.append({"strategy": name, **st})
    monthly = pd.DataFrame(monthly)
    perf = pd.DataFrame(rows).set_index("strategy")
    save_output(monthly, "stage5_monthly_returns")
    save_output(perf, "stage5_performance")
    print("\nPerformance, net of 10 bps (V3_LA_* use future data: illustration only):")
    print(perf.round(3).to_string())

    # --- Does the overlay add value? ------------------------------------------
    sd = pd.DataFrame([{"a": a, "b": "V2", **sharpe_difference_test(monthly[a], monthly["V2"])}
                       for a in ["V3", "V3_R2", "V3_main", "V3_nodegen", "V3_LA_smoothed",
                                 "V3_LA_fullparams"]])
    print("\nSharpe difference vs V2 (stationary bootstrap):")
    print(sd.round(3).to_string(index=False))
    save_output(sd.set_index("a"), "stage5_sharpe_tests")
    span = pd.DataFrame([{"y": a, **spanning_regression(monthly[a], monthly[["V2"]])}
                         for a in ["V3", "V3_R2", "V3_main", "V3_nodegen"]])
    print("\nOverlay on V2 (alpha %/yr, NW t):")
    print(span.round(3).to_string(index=False))
    save_output(span.set_index("y"), "stage5_spanning")

    # --- Is the regime label informative out of sample? ----------------------
    nxt = monthly["V2"]   # month t+1 return, aligned positionally with p at t
    assert (probs.index.to_period("M") + 1 == nxt.index.to_period("M")).all()
    pred = pd.DataFrame([{"model": c, **predictive_test(nxt, probs[c])}
                         for c in ["R1", "R2", "R1_main", "R1_nodegen", "LA_smoothed"]])
    print("\nV2 next-month return on regime probability (slope %/month per unit p, NW t):")
    print(pred.round(3).to_string(index=False))
    save_output(pred.set_index("model"), "stage5_predictive")

    # --- What do the states look like? ---------------------------------------
    for name in ("R1", "R2"):
        all_fits = wf[name]
        df = all_fits[all_fits["degenerate"].eq(False)]
        print(f"\n{name} fitted states over the {len(df)} non-degenerate refits "
              f"(weekly units; median [min, max]):")
        for col in ["mu_hi", "mu_lo", "sd_hi", "sd_lo", "stay_hi", "stay_lo"]:
            print(f"  {col}: {df[col].median():.4f} [{df[col].min():.4f}, {df[col].max():.4f}]")
        print(f"  target state is also the higher-variance state in "
              f"{(df.sd_hi > df.sd_lo).mean():.0%} of these refits")
        print(f"  all {len(all_fits)} months as traded: mean p {all_fits.p.mean():.2f}; "
              f"p > 0.5 in {(all_fits.p > 0.5).mean():.0%}; mean multiplier "
              f"{(REGIME_FLOOR + all_fits.p).mean():.2f}")
    print("\nCorrelation of walk-forward probabilities:")
    print(probs.corr().round(2).to_string())


if __name__ == "__main__":
    main()
