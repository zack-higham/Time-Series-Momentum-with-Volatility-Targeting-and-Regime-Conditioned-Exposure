"""Exposure-normalised regime overlay (pre-registered in config.py, 2026-10-07).

V3 scales V2 by (0.5 + p_t), whose sample mean is 1.19, so it changes both the
level and the timing of exposure. This separates the two:
  V3_norm     V2 x (0.5 + p_t) / (0.5 + pbar_t), pbar_t the expanding mean of
              the R1 walk-forward probabilities up to and including t (tradable)
  V2_x_const  V2 x c, c the realised mean of V3's multiplier (full-sample
              information; a diagnostic, not tradable)
The R1 probabilities are read from Stage 5's output; nothing is refitted.
Writes output/overlay_norm_*.csv; report via run_all to overlay_norm_report.txt.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import monthly_excess, run_backtest
from config import COST_BPS, NORM_OVERLAY_FIRST_LIVE, OUTPUT_DIR, REGIME_FLOOR, SAMPLE_END, newey_west_lags
from data_io import load_frame, save_output
from inference import deflated_sharpe_ratio, sharpe_difference_test, spanning_regression
from metrics import summary
from portfolio import inverse_vol_weights, rebalance_dates, target_portfolio_vol
from returns import daily_excess_returns, daily_returns, load_prices
from riskfree import daily_rf
from signals import tsmom_signal
from volatility import ewma_vol

pd.set_option("display.width", 220)


def predictive_on_multiplier(next_month: pd.Series, m: pd.Series) -> dict:
    """V2 return in month t+1 regressed on m_t (NW t), and Sharpe either side of m_t = 1."""
    df = pd.DataFrame({"r": next_month.values, "m": m.values}).dropna()
    fit = sm.OLS(df["r"], sm.add_constant(df["m"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))})
    hi, lo = df[df.m > 1]["r"], df[df.m <= 1]["r"]
    return {"slope_pct": fit.params["m"] * 100, "t": fit.tvalues["m"],
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

    # --- Multipliers from the saved walk-forward probabilities ----------------
    p = load_frame("stage5_walkforward_R1", OUTPUT_DIR)["p"]
    assert p.index[0] == pd.Timestamp(NORM_OVERLAY_FIRST_LIVE) == dates[0]
    assert p.notna().all() and len(p) == len(dates)
    m_v3 = REGIME_FLOOR + p
    pbar = p.expanding().mean()                 # mean of p_s for s <= t: no future data
    m_norm = m_v3 / (REGIME_FLOOR + pbar)
    c = m_v3.mean()                             # full-sample information (diagnostic only)
    # The expanding mean at t must not change when later probabilities are removed.
    cut = 100
    assert np.allclose(p.iloc[:cut].expanding().mean(), pbar.iloc[:cut])
    print(f"Rebalances: {len(p)} ({p.index[0].date()} to {p.index[-1].date()})")
    print(f"V3 multiplier: mean {m_v3.mean():.4f}, min {m_v3.min():.3f}, max {m_v3.max():.3f}")
    print(f"Normalised multiplier: mean {m_norm.mean():.4f}, min {m_norm.min():.3f}, "
          f"max {m_norm.max():.3f}; share of months > 1: {(m_norm > 1).mean():.2f}")
    print(f"pbar: first {pbar.iloc[0]:.3f}, after 2010-06 {pbar.loc[:'2010-06'].iloc[-1]:.3f}, "
          f"final {pbar.iloc[-1]:.3f}")
    print(f"Constant-leverage multiplier c = {c:.6f} (full-sample information)")
    mult = pd.DataFrame({"p": p, "pbar": pbar, "m_v3": m_v3, "m_norm": m_norm})
    save_output(mult, "overlay_norm_multipliers")

    # --- Backtests: V2 and V3 reproduce Stage 5 exactly ------------------------
    weights = {"V2": w2, "V3": w2.mul(m_v3, axis=0), "V3_norm": w2.mul(m_norm, axis=0),
               "V2_x_const": w2 * c}
    monthly, rows = {}, []
    for name, wt in weights.items():
        res = run_backtest(wt, rets, rf, dates[0], end, cost_bps=COST_BPS)
        monthly[name] = monthly_excess(res.excess_returns, res.rf)
        st = summary(monthly[name], res.excess_returns)
        st["turnover_yr"] = res.turnover.iloc[1:].mean() * 12
        st["mean_gross_lev"] = wt.abs().sum(axis=1).mean()
        rows.append({"strategy": name, **st})
    monthly = pd.DataFrame(monthly)
    s5 = load_frame("stage5_monthly_returns", OUTPUT_DIR)
    for name in ("V2", "V3"):
        assert np.allclose(monthly[name].values, s5[name].values, atol=1e-12), name
    print("V2 and V3 reproduce the Stage 5 monthly returns exactly.")
    perf = pd.DataFrame(rows).set_index("strategy")
    save_output(monthly, "overlay_norm_monthly_returns")
    save_output(perf, "overlay_norm_performance")
    print("\nPerformance, net of 10 bps (V2_x_const uses full-sample information):")
    print(perf.round(3).to_string())

    # --- Sharpe differences (paired stationary bootstrap) ----------------------
    sd = pd.DataFrame([{"a": a, "b": b, **sharpe_difference_test(monthly[a], monthly[b])}
                       for a, b in [("V3_norm", "V2"), ("V2_x_const", "V2"), ("V3", "V2_x_const"),
                                    ("V3", "V3_norm")]])
    print("\nSharpe differences (stationary bootstrap):")
    print(sd.round(3).to_string(index=False))
    save_output(sd.set_index(["a", "b"]), "overlay_norm_sharpe_tests")
    span = pd.DataFrame([{"y": a, **spanning_regression(monthly[a], monthly[["V2"]])}
                         for a in ["V3_norm", "V2_x_const"]])
    print("\nOn V2 (alpha %/yr, NW t):")
    print(span.round(3).to_string(index=False))
    save_output(span.set_index("y"), "overlay_norm_spanning")

    # --- Does the normalised multiplier forecast V2? ---------------------------
    nxt = monthly["V2"]
    assert (m_norm.index.to_period("M") + 1 == nxt.index.to_period("M")).all()
    pred = pd.DataFrame([{"model": "R1_norm", **predictive_on_multiplier(nxt, m_norm)}])
    print("\nV2 next-month return on the normalised multiplier (slope %/month per unit m, NW t):")
    print(pred.round(3).to_string(index=False))
    save_output(pred.set_index("model"), "overlay_norm_predictive")

    # --- Interpretation rule (fixed in config.py before this was run) ---------
    r = sd.set_index(["a", "b"]).loc[("V3_norm", "V2")]
    confirmed = (r["p_value"] >= 0.05) or (r["diff"] < 0)
    print(f"\nRule: V3_norm - V2 = {r['diff']:+.3f}, p = {r['p_value']:.3f} -> "
          f"{'no timing value confirmed' if confirmed else 'TIMING HAS VALUE: revise Section 4.7'}")

    # --- Multiple testing: N = 26 with V3_norm in the trial set ---------------
    srs25 = load_frame("stage6_trial_sharpes", OUTPUT_DIR)["sr_monthly"]
    assert len(srs25) == 25
    m_ = monthly["V3_norm"]
    srs26 = np.append(srs25.values, m_.mean() / m_.std())
    v2 = load_frame("stage6_monthly_returns", OUTPUT_DIR)["V2"]
    assert np.allclose(v2.values, monthly["V2"].values, atol=1e-12)
    d25 = deflated_sharpe_ratio(v2, srs25.values)
    d26 = deflated_sharpe_ratio(v2, srs26)
    dsr = pd.DataFrame([d25, d26]).set_index("n_trials")
    print("\nDeflated Sharpe ratio of V2 (N = 25 as published; N = 26 including V3_norm):")
    print(dsr.round(4).to_string())
    save_output(dsr, "overlay_norm_dsr")


if __name__ == "__main__":
    main()
