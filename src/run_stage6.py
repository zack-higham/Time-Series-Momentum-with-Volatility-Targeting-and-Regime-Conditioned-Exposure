"""Stage 6: full evaluation, crisis alpha, costs, robustness and multiple testing.

Every variant is rebuilt here from the same functions as Stages 3-5, so the
numbers come from one engine. V3 and V3_R2 use the walk-forward regime
probabilities saved by Stage 5 (no refitting here).
Writes output/stage6_*.csv; report via redirect to output/stage6_report.txt.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import monthly_excess, run_backtest
from config import (ASSET_CLASSES, BORROW_FEE_SENSITIVITY_BPS, COST_BPS, COST_SENSITIVITY_BPS,
                    CRISIS_DRAWDOWN_THRESHOLD, FINANCING_SPREAD_SENSITIVITY_BPS,
                    LEVERAGE_CAP_SENSITIVITY, LOOKBACKS_MONTHS, MAIN_START, REGIME_FLOOR,
                    SAMPLE_END, SUBPERIOD_SPLIT, newey_west_lags)
from config import OUTPUT_DIR
from data_io import load_frame, save_output
from inference import deflated_sharpe_ratio, sharpe_difference_test
from metrics import (drawdown_episode, lo_sharpe_se, moments, period_excess, sortino, summary)
from portfolio import (cap_gross_leverage, inverse_vol_weights, portfolio_ex_ante_vol,
                       raw_tsmom_weights, rebalance_dates, target_portfolio_vol)
from returns import daily_excess_returns, daily_returns, load_prices
from riskfree import daily_rf
from signals import all_signals, continuous_tstat_signal
from volatility import ewma_vol

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
MAIN = ["V1", "V2a", "V2", "V3", "V3_R2", "LO_RP", "SPY", "60/40"]
TSMOM = ["V1", "V2a", "V2", "V3", "V3_R2"]


class Ctx:
    """Shared inputs, built once."""

    def __init__(self):
        self.prices = load_prices()
        self.rets = daily_returns(self.prices)
        self.rf = daily_rf(self.prices.index)
        self.dex = daily_excess_returns(self.prices)
        self.vol = ewma_vol(self.dex)
        self.signals = all_signals(self.prices)
        self.dates = rebalance_dates(self.prices.index)
        self.end = pd.Timestamp(SAMPLE_END)

    def v2(self, signal=None, vol=None, allocation="asset_class", exclude_class=None):
        signal = self.signals[12] if signal is None else signal
        vol = self.vol if vol is None else vol
        w, _ = target_portfolio_vol(
            inverse_vol_weights(signal, vol, self.dates, allocation, exclude_class), self.dex, vol)
        return w

    def run(self, w, cost=COST_BPS, **kw):
        res = run_backtest(w, self.rets, self.rf, self.dates[0], self.end, cost_bps=cost, **kw)
        m = monthly_excess(res.excess_returns, res.rf)
        assert len(m) == 222, f"expected 222 months, got {len(m)}"
        return res, m


def regime_weights(w2, column):
    probs = load_frame("stage5_probabilities", OUTPUT_DIR)[column]
    assert probs.index.equals(w2.index), "regime probabilities must align with rebalance dates"
    return w2.mul((REGIME_FLOOR + probs).fillna(1.0), axis=0)


def vol_bias(w, m, ctx, vol=None):
    """Std of monthly return over its ex-ante monthly vol forecast (1 = accurate)."""
    ea = portfolio_ex_ante_vol(w, ctx.dex, ctx.vol if vol is None else vol)
    assert (ea.index.to_period("M") + 1 == m.index.to_period("M")).all()
    return float((m.values / (ea.values / np.sqrt(12))).std(ddof=1))


# --- Crisis windows ------------------------------------------------------------
def drawdown_windows(level: pd.Series, threshold: float) -> pd.DataFrame:
    """Peak-to-trough episodes deeper than threshold. An episode runs from a
    running high to the lowest point before the next new high (or the sample
    end, if the index has not recovered)."""
    rows, peak_date, peak = [], level.index[0], level.iloc[0]
    trough_date, trough = peak_date, peak
    for d, v in level.iloc[1:].items():
        if v >= peak:
            if trough / peak - 1 <= -threshold:
                rows.append((peak_date, trough_date, trough / peak - 1, d))
            peak_date, peak, trough_date, trough = d, v, d, v
        elif v < trough:
            trough_date, trough = d, v
    if trough / peak - 1 <= -threshold:
        rows.append((peak_date, trough_date, trough / peak - 1, None))
    return pd.DataFrame(rows, columns=["peak", "trough", "spy_dd", "recovered"])


def window_return(daily: pd.Series, start, end) -> float:
    """Compounded daily excess return from the day after start through end."""
    x = daily.loc[start:end].iloc[1:]
    return float((1 + x).prod() - 1)


def smile(q_strat: pd.Series, q_spy: pd.Series) -> dict:
    """Quarterly r = a + b SPY + c SPY^2 (Fung and Hsieh 2001), NW t-stats."""
    df = pd.DataFrame({"y": q_strat, "x": q_spy}).dropna()
    X = sm.add_constant(pd.DataFrame({"x": df.x, "x2": df.x ** 2}))
    fit = sm.OLS(df.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(df))})
    return {"a_pct": fit.params["const"] * 100, "b": fit.params["x"], "t_b": fit.tvalues["x"],
            "c": fit.params["x2"], "t_c": fit.tvalues["x2"], "r2": fit.rsquared, "quarters": len(df)}


def main():
    ctx = Ctx()
    s12 = ctx.signals[12]

    # =========================================================================
    # 1. Core variants and metrics
    # =========================================================================
    w = {"V1": raw_tsmom_weights(s12, ctx.dates)}
    w["V2a"] = inverse_vol_weights(s12, ctx.vol, ctx.dates, "asset_class")
    w["V2"] = ctx.v2()
    w["V3"] = regime_weights(w["V2"], "R1")
    w["V3_R2"] = regime_weights(w["V2"], "R2")
    ones = s12.loc[ctx.dates].notna().astype(float)
    w["LO_RP"] = ctx.v2(signal=ones)
    zeros = pd.DataFrame(0.0, index=ctx.dates, columns=s12.columns)
    w["SPY"] = zeros.assign(SPY=1.0)
    w["60/40"] = zeros.assign(SPY=0.6, IEF=0.4)

    res, monthly, daily = {}, {}, {}
    for n, wt in w.items():
        res[n], monthly[n] = ctx.run(wt, cost=0.0 if n in ("SPY", "60/40") else COST_BPS)
        daily[n] = res[n].excess_returns
    monthly, daily = pd.DataFrame(monthly), pd.DataFrame(daily)
    quarterly = pd.DataFrame({n: period_excess(daily[n], ctx.rf, "Q") for n in MAIN})
    assert len(quarterly) == 74 and str(quarterly.index[0]) == "2008Q2"

    # Cross-check against Stage 4/5 outputs: identical engine, identical numbers.
    s4 = load_frame("stage4_monthly_returns", OUTPUT_DIR)
    s5 = load_frame("stage5_monthly_returns", OUTPUT_DIR)
    for n in ("V1", "V2a", "V2", "LO_RP", "SPY", "60/40"):
        assert np.allclose(monthly[n].values, s4[n].values), f"{n} differs from Stage 4"
    for n in ("V3", "V3_R2"):
        assert np.allclose(monthly[n].values, s5[n].values), f"{n} differs from Stage 5"
    print("Cross-check: all core monthly series match Stages 4-5 exactly.")

    rows = []
    for n in MAIN:
        st = summary(monthly[n], daily[n])
        st["sharpe_se_lo"] = lo_sharpe_se(monthly[n])
        st["sortino"] = sortino(monthly[n])
        st.update(drawdown_episode(daily[n]))
        for label, x in (("d", daily[n]), ("m", monthly[n]), ("q", quarterly[n])):
            mo = moments(x)
            st[f"skew_{label}"], st[f"skew_se_{label}"] = mo["skew"], mo["skew_se"]
            st[f"exkurt_{label}"], st[f"exkurt_se_{label}"] = mo["exkurt"], mo["exkurt_se"]
        st["turnover_yr"] = res[n].turnover.iloc[1:].mean() * 12
        st["gross_lev_mean"] = w[n].abs().sum(axis=1).mean()
        rows.append({"strategy": n, **st})
    core = pd.DataFrame(rows).set_index("strategy")
    save_output(core, "stage6_core_metrics")
    save_output(monthly, "stage6_monthly_returns")
    save_output(daily, "stage6_daily_returns")
    save_output(quarterly.set_axis(quarterly.index.astype(str)), "stage6_quarterly_returns")
    print("\n1. Core metrics, 2008-04 to 2026-09 (TSMOM and LO_RP net of 10 bps):")
    print(core[["ann_ret_pct", "ann_vol_pct", "sharpe", "sharpe_se_lo", "nw_t", "sortino",
                "max_dd_pct", "calmar", "mdd_peak", "mdd_trough", "mdd_recovery",
                "mdd_underwater_m", "longest_underwater_m", "hit_rate"]].round(3).to_string())
    print("\nHigher moments (skew, excess kurtosis; normal-theory SEs in skew_se_*):")
    print(core[["skew_d", "skew_m", "skew_q", "skew_se_d", "skew_se_m", "skew_se_q",
                "exkurt_d", "exkurt_m", "exkurt_q"]].round(3).to_string())

    # =========================================================================
    # 2. Crisis alpha
    # =========================================================================
    spy_level = (1 + ctx.rets["SPY"].loc[ctx.dates[0]:ctx.end].iloc[1:]).cumprod()
    spy_level = pd.concat([pd.Series(1.0, index=[ctx.dates[0]]), spy_level])
    win = drawdown_windows(spy_level, CRISIS_DRAWDOWN_THRESHOLD)
    for n in MAIN:
        win[n] = [window_return(daily[n], r.peak, r.trough) * 100 for r in win.itertuples()]
    win["days"] = [(r.trough - r.peak).days for r in win.itertuples()]
    print(f"\n2. Crisis windows: SPY total-return drawdowns deeper than "
          f"{CRISIS_DRAWDOWN_THRESHOLD:.0%} (peak to trough), cumulative excess return %:")
    print(win.round(2).to_string(index=False))
    save_output(win.set_index("peak"), "stage6_crisis_windows")
    # Recovery legs: trough to the next SPY high (or sample end): does trend give it back?
    rec = []
    for r in win.itertuples():
        stop = ctx.end if pd.isna(r.recovered) else r.recovered
        rec.append({"trough": r.trough, "to": stop,
                    **{n: window_return(daily[n], r.trough, stop) * 100 for n in MAIN}})
    rec = pd.DataFrame(rec)
    print("Recovery legs (trough to next SPY high), cumulative excess return %:")
    print(rec.round(2).to_string(index=False))
    save_output(rec.set_index("trough"), "stage6_crisis_recoveries")

    cond = []
    for label, frame in (("month", monthly), ("quarter", quarterly)):
        cut = frame["SPY"].quantile(0.10)
        worst = frame["SPY"] <= cut
        for n in MAIN:
            cond.append({"freq": label, "strategy": n, "n_worst": int(worst.sum()),
                         "mean_worst_pct": frame.loc[worst, n].mean() * 100,
                         "hit_worst": (frame.loc[worst, n] > 0).mean(),
                         "mean_rest_pct": frame.loc[~worst, n].mean() * 100})
    cond = pd.DataFrame(cond)
    print("\nConditional on SPY's worst-decile months / quarters (mean %):")
    print(cond.pivot(index="strategy", columns="freq",
                     values=["mean_worst_pct", "hit_worst", "mean_rest_pct"]).loc[MAIN]
          .round(2).to_string())
    save_output(cond.set_index(["freq", "strategy"]), "stage6_worst_decile")

    # =========================================================================
    # 3. Trend smile
    # =========================================================================
    sm_rows = [{"strategy": n, **smile(quarterly[n], quarterly["SPY"])} for n in MAIN if n != "SPY"]
    sm_tab = pd.DataFrame(sm_rows).set_index("strategy")
    print("\n3. Smile: quarterly r = a + b SPY + c SPY^2 (NW t):")
    print(sm_tab.round(3).to_string())
    save_output(sm_tab, "stage6_smile")

    # =========================================================================
    # 4. Costs, borrow and financing
    # =========================================================================
    cost_rows = []
    for n in TSMOM + ["LO_RP"]:
        gross_res, gross_m = ctx.run(w[n], cost=0.0)
        g = summary(gross_m, gross_res.excess_returns)
        turn = gross_res.turnover.iloc[1:].mean() * 12
        row = {"strategy": n, "turnover_yr": turn, "gross_ret_pct": g["ann_ret_pct"],
               "gross_sharpe": g["sharpe"],
               # Linear break-even: annual gross mean / annual turnover (cost per unit traded).
               "breakeven_bps": g["ann_ret_pct"] / 100 / turn * 1e4}
        for c in COST_SENSITIVITY_BPS:
            if c == 0:
                row["sharpe_0"] = g["sharpe"]
                continue
            _, mc = ctx.run(w[n], cost=c)
            row[f"sharpe_{c}"] = mc.mean() / mc.std() * np.sqrt(12)
        cost_rows.append(row)
    costs = pd.DataFrame(cost_rows).set_index("strategy")
    print("\n4. Cost sensitivity (Sharpe by one-way cost in bps; break-even in bps per unit traded):")
    print(costs.round(3).to_string())
    save_output(costs, "stage6_costs")

    fee_rows = []
    for n in ("V2", "V3", "LO_RP"):
        for kind, levels in (("borrow", BORROW_FEE_SENSITIVITY_BPS),
                             ("financing", FINANCING_SPREAD_SENSITIVITY_BPS)):
            for b in levels:
                kw = {"borrow_bps": b} if kind == "borrow" else {"financing_bps": b}
                r_, m_ = ctx.run(w[n], **kw)
                fee_rows.append({"strategy": n, "fee": kind, "bps": b,
                                 "ann_ret_pct": m_.mean() * 1200,
                                 "sharpe": m_.mean() / m_.std() * np.sqrt(12)})
        r_, m_ = ctx.run(w[n], borrow_bps=100, financing_bps=100)
        fee_rows.append({"strategy": n, "fee": "both", "bps": 100, "ann_ret_pct": m_.mean() * 1200,
                         "sharpe": m_.mean() / m_.std() * np.sqrt(12)})
    fees = pd.DataFrame(fee_rows)
    lev = w["V2"]
    print("Borrow fee (on short notional) and financing spread (on borrowed cash), net of 10 bps:")
    print(fees.round(3).to_string(index=False))
    print(f"V2 average short notional {lev.clip(upper=0).abs().sum(axis=1).mean():.2f}, "
          f"average borrowed cash {(lev.clip(lower=0).sum(axis=1) - 1).clip(lower=0).mean():.2f} x NAV")
    save_output(fees.set_index(["strategy", "fee", "bps"]), "stage6_fees")

    # =========================================================================
    # 5. Robustness grid (each reported, none selected)
    # =========================================================================
    rob_w, rob_kw, rob_vol = {}, {}, {}
    rob_w["primary (V2)"] = w["V2"]
    rob_w["execution lag 1 day"], rob_kw["execution lag 1 day"] = w["V2"], {"lag": 1}
    for com in (20, 120):
        v = ewma_vol(ctx.dex, com=com)
        rob_w[f"EWMA com {com}"] = ctx.v2(vol=v)
        rob_vol[f"EWMA com {com}"] = v
    rob_w["equal risk per instrument"] = ctx.v2(allocation="instrument")
    for cap in LEVERAGE_CAP_SENSITIVITY:
        rob_w[f"gross leverage cap {cap:g}x"] = cap_gross_leverage(w["V2"], cap)
    cont = continuous_tstat_signal(ctx.dex, ctx.dates)
    assert cont.notna().all().all(), "continuous signal missing in main sample"
    rob_w["continuous t-stat signal"] = ctx.v2(signal=cont.reindex(s12.index))
    for k in LOOKBACKS_MONTHS:
        if k != 12:
            rob_w[f"lookback {k}m"] = ctx.v2(signal=ctx.signals[k])
    rob_w["blend of 1/3/6/12m"] = ctx.v2(signal=ctx.signals["blend"])
    for c in ASSET_CLASSES:
        rob_w[f"excluding {c}"] = ctx.v2(exclude_class=c)

    rob_rows, rob_monthly = [], {}
    for name, wt in rob_w.items():
        r_, m_ = ctx.run(wt, **rob_kw.get(name, {}))
        rob_monthly[name] = m_
        st = summary(m_, r_.excess_returns)
        sd = sharpe_difference_test(m_, monthly["V2"]) if name != "primary (V2)" else {}
        rob_rows.append({"variant": name, **{k: st[k] for k in
                                             ("ann_ret_pct", "ann_vol_pct", "sharpe", "nw_t",
                                              "max_dd_pct", "skew_m")},
                         "sharpe_se_lo": lo_sharpe_se(m_),
                         "diff_vs_V2": sd.get("diff", 0.0), "p_vs_V2": sd.get("p_value", np.nan),
                         "corr_V2": m_.corr(monthly["V2"]),
                         "turnover_yr": r_.turnover.iloc[1:].mean() * 12,
                         "gross_lev_mean": wt.abs().sum(axis=1).mean(),
                         "vol_bias": vol_bias(wt, m_, ctx, rob_vol.get(name))})
    rob = pd.DataFrame(rob_rows).set_index("variant")
    print("\n5. Robustness grid (V2 definition unless stated; net of 10 bps; vol_bias = "
          "std of monthly return / ex-ante monthly vol):")
    print(rob.round(3).to_string())
    save_output(rob, "stage6_robustness")
    save_output(pd.DataFrame(rob_monthly), "stage6_robustness_monthly")

    # Subperiod halves
    half = []
    split = pd.Period(SUBPERIOD_SPLIT, "M")
    per = monthly.index.to_period("M")
    for label, mask in (("2008-04 to 2017-06", per < split), ("2017-07 to 2026-09", per >= split)):
        for n in MAIN:
            x = monthly.loc[mask, n]
            half.append({"half": label, "strategy": n, "months": len(x),
                         "ann_ret_pct": x.mean() * 1200, "ann_vol_pct": x.std() * np.sqrt(12) * 100,
                         "sharpe": x.mean() / x.std() * np.sqrt(12), "sharpe_se_lo": lo_sharpe_se(x)})
    half = pd.DataFrame(half)
    print("\nSubperiod halves (Sharpe, Lo SE):")
    print(half.pivot(index="strategy", columns="half", values=["sharpe", "sharpe_se_lo",
                                                               "ann_ret_pct"]).loc[MAIN]
          .round(3).to_string())
    save_output(half.set_index(["half", "strategy"]), "stage6_subperiods")

    # =========================================================================
    # 6. Multiple testing: every TSMOM configuration examined in Stages 3-6
    # =========================================================================
    # Stage 4's lookback and per-instrument variants are rows of the robustness grid.
    trials = {"V1": monthly["V1"], "V2a": monthly["V2a"]}
    s5m = load_frame("stage5_monthly_returns", OUTPUT_DIR)
    for n in ("V3", "V3_R2", "V3_main", "V3_nodegen"):
        trials[n] = s5m[n]
    for name, m_ in rob_monthly.items():
        trials[f"rob: {name}"] = m_
    for k in LOOKBACKS_MONTHS:   # Stage 3: raw TSMOM at each lookback
        if k != 12:
            _, m_ = ctx.run(raw_tsmom_weights(ctx.signals[k], ctx.dates))
            trials[f"V1_k{k}"] = m_
    tr = pd.DataFrame({k: pd.Series(v.values) for k, v in trials.items()})
    # Drop exact duplicates (the 12m robustness row is V2 itself, etc.).
    tr = tr.T.drop_duplicates().T
    srs = (tr.mean() / tr.std()).values
    dsr = deflated_sharpe_ratio(monthly["V2"], srs)
    print(f"\n6. Multiple testing: {dsr['n_trials']} distinct TSMOM configurations examined.")
    print({k: round(float(v), 3) for k, v in dsr.items()})
    # Trials are highly correlated, so N overstates the number of independent
    # tests: report DSR across a range of effective N as well.
    eff = []
    for n_eff in (2, 5, 10, 20, dsr["n_trials"]):
        d = deflated_sharpe_ratio(monthly["V2"], srs, n_trials=n_eff)
        eff.append({"n_trials": n_eff, "sr0_ann": d["sr0_ann"], "dsr": d["dsr"]})
    eff = pd.DataFrame(eff)
    print("DSR sensitivity to the effective number of trials (cross-trial SR sd held fixed):")
    print(eff.round(3).to_string(index=False))
    save_output(pd.DataFrame([dsr]), "stage6_deflated_sharpe")
    save_output(eff.set_index("n_trials"), "stage6_dsr_sensitivity")
    save_output(pd.Series(tr.columns, name="trial"), "stage6_trials")
    save_output(pd.Series(srs, index=tr.columns, name="sr_monthly"), "stage6_trial_sharpes")


if __name__ == "__main__":
    main()
