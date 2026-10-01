"""Stage 4: volatility targeting (Variant 2), benchmarks and spanning tests.

Strategies (12-month signal unless stated; TSMOM and risk parity net of 10 bps):
  V1        raw TSMOM, equal notional (Stage 3)
  V2a       inverse-vol weights, equal risk per asset class, no portfolio scaling
  V2        V2a scaled to 10% ex-ante portfolio volatility (primary)
  V2_inst   as V2 but equal risk per instrument (MOP convention; robustness)
  LO_RP     long-only risk parity: V2 with every signal set to +1
  SPY, 60/40 (SPY/IEF, monthly rebalanced): passive benchmarks, no costs
Writes output/stage4_*.csv and output/stage4_report.txt (via tee).
"""

import numpy as np
import pandas as pd

from backtest import monthly_excess, run_backtest
from config import (ASSET_CLASSES, COST_BPS, INSTRUMENTS, LOOKBACKS_MONTHS, MAIN_START,
                    PORTFOLIO_VOL_TARGET, SAMPLE_END)
from data_io import load_frame, save_output
from inference import sharpe_difference_test, spanning_regression
from metrics import summary
from portfolio import (inverse_vol_weights, portfolio_ex_ante_vol, raw_tsmom_weights,
                       rebalance_dates, risk_contributions, target_portfolio_vol)
from returns import daily_excess_returns, daily_returns, load_prices
from riskfree import daily_rf
from signals import all_signals
from volatility import ewma_vol

pd.set_option("display.width", 220)
CLASSES = pd.Series(INSTRUMENTS)


def class_sleeves(result, daily_excess):
    """Monthly return contribution of each asset class: sum over the month of
    w_{d-1} (r_d - rf_d). Additive (not compounded); used for correlations."""
    contrib = result.weights.shift(1) * daily_excess.loc[result.weights.index, result.weights.columns]
    by_class = contrib.T.groupby(CLASSES).sum().T
    return by_class.groupby(by_class.index.to_period("M")).sum()   # PeriodIndex


def main():
    prices = load_prices()
    rets, rf = daily_returns(prices), daily_rf(prices.index)
    dex = daily_excess_returns(prices)
    vol = ewma_vol(dex)
    signals = all_signals(prices)
    dates = rebalance_dates(prices.index)
    end = pd.Timestamp(SAMPLE_END)
    s12 = signals[12]

    # --- Weights ------------------------------------------------------------
    w = {}
    w["V1"] = raw_tsmom_weights(s12, dates)
    w["V2a"] = inverse_vol_weights(s12, vol, dates, "asset_class")
    w["V2"], scale_v2 = target_portfolio_vol(w["V2a"], dex, vol)
    w["V2_inst"], _ = target_portfolio_vol(inverse_vol_weights(s12, vol, dates, "instrument"), dex, vol)
    ones = s12.loc[dates].notna().astype(float)
    w["LO_RP"], _ = target_portfolio_vol(inverse_vol_weights(ones, vol, dates, "asset_class"), dex, vol)
    zeros = pd.DataFrame(0.0, index=dates, columns=s12.columns)
    w["SPY"] = zeros.assign(SPY=1.0)
    w["60/40"] = zeros.assign(SPY=0.6, IEF=0.4)
    for name, wt in w.items():
        assert wt.notna().all().all(), f"missing weight in {name}"

    # --- Backtests ----------------------------------------------------------
    results, monthly = {}, {}
    for name, wt in w.items():
        cost = 0.0 if name in ("SPY", "60/40") else COST_BPS
        res = run_backtest(wt, rets, rf, dates[0], end, cost_bps=cost)
        results[name] = res
        monthly[name] = monthly_excess(res.excess_returns, res.rf)
        assert len(monthly[name]) == 222
    monthly = pd.DataFrame(monthly)

    rows = []
    for name, res in results.items():
        st = summary(monthly[name], res.excess_returns)
        st["turnover_yr"] = res.turnover.iloc[1:].mean() * 12
        gross = w[name].abs().sum(axis=1)
        st["gross_lev_mean"] = gross.mean()
        st["gross_lev_max"] = gross.max()
        rows.append({"strategy": name, **st})
    perf = pd.DataFrame(rows).set_index("strategy")
    save_output(perf, "stage4_performance")
    save_output(monthly, "stage4_monthly_returns")
    save_output(pd.DataFrame({k: r.excess_returns for k, r in results.items()}), "stage4_daily_returns")
    print("Performance, 2008-04 to 2026-09 (TSMOM and LO_RP net of 10 bps):")
    print(perf.round(3).to_string())

    # --- Vol-targeting validation -------------------------------------------
    print("\nVol targeting validation:")
    ex_ante = {n: portfolio_ex_ante_vol(w[n], dex, vol) for n in ("V1", "V2a", "V2", "LO_RP")}
    val_rows = []
    for n, ea in ex_ante.items():
        m = monthly[n]
        # The forecast made at rebalance date t applies to the following month:
        # 222 rebalances (2008-03..2026-08) map one-to-one onto 222 months
        # (2008-04..2026-09).
        assert len(ea) == len(m) and (ea.index.to_period("M") + 1 == m.index.to_period("M")).all()
        ea_hold = pd.Series(ea.values, index=m.index)
        z = m / (ea_hold / np.sqrt(12))
        realised = m.std() * np.sqrt(12)
        val_rows.append({"strategy": n, "mean_ex_ante_pct": ea.mean() * 100,
                         "realised_pct": realised * 100, "bias_stat": z.std(),
                         "months_abs_z_gt_2": int((z.abs() > 2).sum()),
                         "months_abs_z_gt_3": int((z.abs() > 3).sum())})
    val = pd.DataFrame(val_rows).set_index("strategy")
    print(val.round(3).to_string())
    roll = monthly[["V1", "V2a", "V2"]].rolling(12).std() * np.sqrt(12) * 100
    print("Rolling 12m realised vol (%), V2: min {:.1f}, median {:.1f}, max {:.1f}".format(
        roll["V2"].min(), roll["V2"].median(), roll["V2"].max()))
    save_output(val, "stage4_vol_validation")
    save_output(roll, "stage4_rolling_vol")

    lev = w["V2"].abs().sum(axis=1)
    net = w["V2"].sum(axis=1)
    lev_stats = {"mean": lev.mean(), "median": lev.median(), "p95": lev.quantile(0.95),
                 "max": lev.max(), "max_date": lev.idxmax().date(),
                 "share_gt_2": (lev > 2).mean(), "share_gt_3": (lev > 3).mean(),
                 "net_mean": net.mean()}
    print("V2 gross leverage at rebalance:", {k: (round(v, 3) if isinstance(v, float) else v)
                                              for k, v in lev_stats.items()})
    print("V2 vol-scaling factor (2a -> 2b): min {:.2f}, median {:.2f}, max {:.2f}".format(
        scale_v2.min(), scale_v2.median(), scale_v2.max()))
    save_output(pd.DataFrame({"gross": lev, "net": net, "scale": scale_v2}), "stage4_leverage")

    # --- Risk contributions by asset class ----------------------------------
    rc_rows = {}
    for n in ("V1", "V2", "V2_inst"):
        rc = risk_contributions(w[n], dex, vol)
        rc_rows[n] = rc.T.groupby(CLASSES).sum().T.mean()
    rc_table = pd.DataFrame(rc_rows).loc[list(ASSET_CLASSES)]
    print("\nAverage ex-ante share of portfolio variance by asset class:")
    print(rc_table.round(3).to_string())
    save_output(rc_table, "stage4_risk_contributions")

    # --- Horizons, vol-targeted ---------------------------------------------
    hz_rows = []
    for k in list(LOOKBACKS_MONTHS) + ["blend"]:
        wk, _ = target_portfolio_vol(inverse_vol_weights(signals[k], vol, dates, "asset_class"), dex, vol)
        res = run_backtest(wk, rets, rf, dates[0], end, cost_bps=COST_BPS)
        mk = monthly_excess(res.excess_returns, res.rf)
        hz_rows.append({"lookback": k, **summary(mk, res.excess_returns),
                        "turnover_yr": res.turnover.iloc[1:].mean() * 12})
        monthly[f"V2_k{k}"] = mk
    hz = pd.DataFrame(hz_rows).set_index("lookback")
    print("\nV2 by lookback (net of 10 bps):")
    print(hz.round(3).to_string())
    save_output(hz, "stage4_horizons")

    # --- Spanning regressions ------------------------------------------------
    span_rows = []
    specs = {"SPY": ["SPY"], "60/40": ["60/40"], "LO_RP": ["LO_RP"], "SPY+LO_RP": ["SPY", "LO_RP"]}
    for y in ("V1", "V2"):
        for label, cols in specs.items():
            span_rows.append({"y": y, "factors": label,
                              **spanning_regression(monthly[y], monthly[cols])})
    span = pd.DataFrame(span_rows)
    print("\nSpanning regressions (alpha %/yr, NW t):")
    print(span.round(3).to_string(index=False))
    save_output(span.set_index(["y", "factors"]), "stage4_spanning")

    # --- Sharpe-difference tests --------------------------------------------
    pairs = [("V2", "V1"), ("V2a", "V1"), ("V2", "V2a"), ("V2", "LO_RP"), ("V2", "V2_inst")]
    sd = pd.DataFrame([{"a": a, "b": b, **sharpe_difference_test(monthly[a], monthly[b])}
                       for a, b in pairs])
    print("\nSharpe-difference tests (stationary bootstrap, 5,000 reps, mean block 6 months):")
    print(sd.round(3).to_string(index=False))
    save_output(sd.set_index(["a", "b"]), "stage4_sharpe_tests")

    # --- External validation against AQR ------------------------------------
    aqr = load_frame("aqr_tsmom").loc[MAIN_START:]
    # AQR dates are calendar month-ends, mine are last trading days: align on
    # the calendar month (a date join silently drops weekend month-ends).
    aqr.index = aqr.index.to_period("M")
    mp = monthly.copy()
    mp.index = mp.index.to_period("M")
    common = mp.index.intersection(aqr.index)
    assert len(common) == len(aqr), "every AQR month in the sample should match"
    print(f"\nCorrelation with AQR TSMOM ({len(common)} months, {common[0]} to {common[-1]}):")
    print(mp.loc[common, ["V1", "V2a", "V2", "V2_inst", "LO_RP", "SPY"]]
          .corrwith(aqr.loc[common, "tsmom"]).round(3).to_string())
    sleeves = class_sleeves(results["V2"], dex)
    aqr_map = {"Equity": "tsmom_eq", "Rates": "tsmom_fi", "Commodity": "tsmom_cm", "FX": "tsmom_fx"}
    sleeve_corr = {c: sleeves.loc[common, c].corr(aqr.loc[common, a]) for c, a in aqr_map.items()}
    print("V2 asset-class sleeves vs AQR asset-class factors:",
          {k: round(v, 3) for k, v in sleeve_corr.items()})
    aqr_span = spanning_regression(mp.loc[common, "V2"], aqr.loc[common, ["tsmom"]])
    print("V2 on AQR TSMOM:", {k: round(v, 3) for k, v in aqr_span.items()})
    save_output(pd.Series(sleeve_corr, name="corr"), "stage4_aqr_sleeve_corr")


if __name__ == "__main__":
    main()
