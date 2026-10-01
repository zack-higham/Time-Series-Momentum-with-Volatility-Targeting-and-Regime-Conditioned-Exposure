"""Stage 13: phase 2 portfolios and evaluation.

Strategies (all net of 10 bps; vol-targeted to 10% ex ante as in V2):
  V2          phase 1 primary (10 ETFs, trend)                     [reference]
  TREND_U2    V2 rules on the 34-ETF universe                      (breadth)
  CARRY_U2    cross-sectional carry, equal risk per carry class    (new premium)
  COMBO_U2    0.5 TREND_U2 + 0.5 CARRY_U2, rescaled to 10%         PRIMARY
  COMBO_CORE  the same combination on the 10 core ETFs             (signal diversification
                                                                    without breadth)
Robustness (reported, none selected): time-series carry; one-day execution
lag; 3x gross leverage cap; carry without EWT's two capital-gains
distributions (diagnostic). Portfolio value: 60/40 blended with COMBO_U2 at
fixed weights, monthly rebalanced.
Writes output/stage13_*.csv; report via redirect to output/stage13_report.txt.
"""

import numpy as np
import pandas as pd

from backtest import monthly_excess, run_backtest
from carry import carry_panel, rank_signal, sign_signal
from config import (BLEND_WEIGHTS, CARRY_CLASSES, COMBO_RISK_WEIGHTS, COST_BPS, COST_SENSITIVITY_BPS,
                    CRISIS_DRAWDOWN_THRESHOLD, INSTRUMENTS, MAIN_START, OUTPUT_DIR, SAMPLE_END,
                    UNIVERSE_U2)
from data_io import load_frame, save_output
from inference import sharpe_difference_test, spanning_regression
from metrics import drawdown_episode, lo_sharpe_se, moments, period_excess, sortino, summary
from portfolio import (cap_gross_leverage, inverse_vol_weights, portfolio_ex_ante_vol,
                       rebalance_dates, target_portfolio_vol)
from returns import daily_excess_returns, daily_returns, load_prices_u2
from riskfree import daily_rf
from run_stage6 import drawdown_windows, smile, window_return
from run_stage12 import EWT_CG_DATES
from signals import tsmom_signal
from volatility import ewma_vol

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
CORE = list(INSTRUMENTS)


def carry_classes(universe):
    return {t: c for t, c in universe.items() if c in CARRY_CLASSES}


class P2:
    def __init__(self):
        self.prices = load_prices_u2()
        self.rets = daily_returns(self.prices)
        self.rf = daily_rf(self.prices.index)
        self.dex = daily_excess_returns(self.prices)
        self.vol = ewma_vol(self.dex)
        self.dates = rebalance_dates(self.prices.index)
        self.end = pd.Timestamp(SAMPLE_END)
        close = pd.concat([load_frame("close"), load_frame("u2add_close")], axis=1)[list(UNIVERSE_U2)]
        self.actions = pd.concat([load_frame("actions"), load_frame("u2add_actions")]).sort_index()
        self.close = close
        self.trend_sig = tsmom_signal(self.prices, 12)

    def carry(self, actions=None):
        return carry_panel(UNIVERSE_U2, self.close, self.actions if actions is None else actions,
                           self.prices.index)

    def targeted(self, signal, classes):
        """Inverse-vol with equal risk per class over `classes`, scaled to 10%."""
        w = inverse_vol_weights(signal.reindex(columns=list(classes)), self.vol, self.dates,
                                "asset_class", classes=classes)
        w, _ = target_portfolio_vol(w.fillna(0.0), self.dex, self.vol)
        return w.reindex(columns=list(UNIVERSE_U2), fill_value=0.0)

    def combine(self, w_trend, w_carry):
        w = COMBO_RISK_WEIGHTS["trend"] * w_trend + COMBO_RISK_WEIGHTS["carry"] * w_carry
        w, _ = target_portfolio_vol(w, self.dex, self.vol)
        return w

    def run(self, w, cost=COST_BPS, **kw):
        res = run_backtest(w, self.rets, self.rf, self.dates[0], self.end, cost_bps=cost, **kw)
        m = monthly_excess(res.excess_returns, res.rf)
        assert len(m) == 222
        return res, m


def build_weights(p: P2) -> dict:
    u2 = UNIVERSE_U2
    core = {t: u2[t] for t in CORE}
    carry = p.carry()
    w = {}
    w["V2"] = p.targeted(p.trend_sig, core)
    w["TREND_U2"] = p.targeted(p.trend_sig, u2)
    w["CARRY_U2"] = p.targeted(rank_signal(carry, u2), carry_classes(u2))
    w["COMBO_U2"] = p.combine(w["TREND_U2"], w["CARRY_U2"])
    core_carry = rank_signal(carry[[t for t in CORE if t in carry.columns]], core)
    w["CARRY_CORE"] = p.targeted(core_carry, carry_classes(core))
    w["COMBO_CORE"] = p.combine(w["V2"], w["CARRY_CORE"])
    # Robustness and diagnostic variants of the primary
    w["CARRY_TS"] = p.targeted(sign_signal(carry), carry_classes(u2))
    w["COMBO_TS"] = p.combine(w["TREND_U2"], w["CARRY_TS"])
    w["COMBO_CAP3"] = cap_gross_leverage(w["COMBO_U2"], 3.0)
    drop = p.actions.index.isin(pd.to_datetime(EWT_CG_DATES)) & (p.actions["ticker"] == "EWT")
    carry_d = p.carry(p.actions[~drop])
    w["CARRY_EWTDIAG"] = p.targeted(rank_signal(carry_d, u2), carry_classes(u2))
    w["COMBO_EWTDIAG"] = p.combine(w["TREND_U2"], w["CARRY_EWTDIAG"])
    zeros = pd.DataFrame(0.0, index=p.dates, columns=list(u2))
    w["SPY"] = zeros.assign(SPY=1.0)
    w["60/40"] = zeros.assign(SPY=0.6, IEF=0.4)
    for n, x in w.items():
        assert x.notna().all().all(), n
    return w


def main():
    p = P2()
    w = build_weights(p)
    res, monthly, daily = {}, {}, {}
    for n, x in w.items():
        res[n], monthly[n] = p.run(x, cost=0.0 if n in ("SPY", "60/40") else COST_BPS)
        daily[n] = res[n].excess_returns
    res["COMBO_LAG1"], monthly["COMBO_LAG1"] = p.run(w["COMBO_U2"], lag=1)
    daily["COMBO_LAG1"] = res["COMBO_LAG1"].excess_returns
    monthly, daily = pd.DataFrame(monthly), pd.DataFrame(daily)

    # Cross-check: V2 rebuilt from the U2 panel equals phase 1's V2 exactly.
    s6 = load_frame("stage6_monthly_returns", OUTPUT_DIR)
    assert np.allclose(monthly["V2"].values, s6["V2"].values), "V2 differs from phase 1"
    print("Cross-check: V2 rebuilt on the U2 panel matches phase 1 exactly.")

    main_names = ["V2", "TREND_U2", "CARRY_U2", "COMBO_U2", "CARRY_CORE", "COMBO_CORE", "SPY", "60/40"]
    rob_names = ["COMBO_TS", "CARRY_TS", "COMBO_LAG1", "COMBO_CAP3", "CARRY_EWTDIAG", "COMBO_EWTDIAG"]
    quarterly = pd.DataFrame({n: period_excess(daily[n], p.rf, "Q") for n in main_names})
    rows = []
    for n in main_names + rob_names:
        st = summary(monthly[n], daily[n])
        st["sharpe_se_lo"] = lo_sharpe_se(monthly[n])
        st["sortino"] = sortino(monthly[n])
        st.update(drawdown_episode(daily[n]))
        st["skew_q"] = moments(quarterly[n])["skew"] if n in quarterly else np.nan
        st["turnover_yr"] = res[n].turnover.iloc[1:].mean() * 12
        wt = w["COMBO_U2"] if n == "COMBO_LAG1" else w[n]
        st["gross_lev_mean"] = wt.abs().sum(axis=1).mean()
        st["gross_lev_max"] = wt.abs().sum(axis=1).max()
        rows.append({"strategy": n, **st})
    perf = pd.DataFrame(rows).set_index("strategy")
    save_output(perf, "stage13_performance")
    save_output(monthly, "stage13_monthly_returns")
    save_output(daily, "stage13_daily_returns")
    print("\n1. Phase 2 performance, 2008-04 to 2026-09 (net of 10 bps; SPY, 60/40 gross):")
    print(perf[["ann_ret_pct", "ann_vol_pct", "sharpe", "sharpe_se_lo", "nw_t", "sortino",
                "max_dd_pct", "calmar", "skew_m", "skew_q", "turnover_yr", "gross_lev_mean",
                "gross_lev_max", "mdd_peak", "mdd_trough", "mdd_recovery"]].round(3).to_string())

    # --- Tests ---------------------------------------------------------------------
    pairs = [("COMBO_U2", "V2"), ("TREND_U2", "V2"), ("COMBO_U2", "TREND_U2"),
             ("COMBO_CORE", "V2"), ("COMBO_U2", "COMBO_CORE"), ("COMBO_U2", "60/40")]
    sd = pd.DataFrame([{"a": a, "b": b, **sharpe_difference_test(monthly[a], monthly[b])}
                       for a, b in pairs])
    print("\n2. Sharpe-difference tests (stationary bootstrap):")
    print(sd.round(3).to_string(index=False))
    save_output(sd.set_index(["a", "b"]), "stage13_sharpe_tests")
    corr = monthly[["V2", "TREND_U2", "CARRY_U2", "COMBO_U2", "CARRY_CORE", "SPY", "60/40"]].corr()
    print("\nMonthly correlations:")
    print(corr.round(2).to_string())
    save_output(corr, "stage13_correlations")
    span = pd.DataFrame([{"y": y, "factors": "+".join(f), **spanning_regression(monthly[y], monthly[f])}
                         for y, f in [("CARRY_U2", ["TREND_U2"]), ("COMBO_U2", ["V2"]),
                                      ("COMBO_U2", ["SPY"]), ("COMBO_U2", ["60/40"])]])
    print("\nSpanning regressions (alpha %/yr, NW t):")
    print(span.round(3).to_string(index=False))
    save_output(span.set_index(["y", "factors"]), "stage13_spanning")
    aqr = load_frame("aqr_tsmom").loc[MAIN_START:]
    mp = monthly.copy()
    mp.index, aqr.index = mp.index.to_period("M"), aqr.index.to_period("M")
    common = mp.index.intersection(aqr.index)
    assert len(common) == len(aqr)
    print("Correlation with AQR TSMOM (218 months):",
          mp.loc[common, ["V2", "TREND_U2", "CARRY_U2", "COMBO_U2"]].corrwith(aqr.loc[common, "tsmom"])
          .round(3).to_dict())

    # --- Calendar years and halves ------------------------------------------------
    yearly = (1 + monthly[["V2", "TREND_U2", "CARRY_U2", "COMBO_U2", "SPY"]]).groupby(
        monthly.index.year).prod() - 1
    print("\nCalendar-year excess returns (%):")
    print((yearly * 100).round(1).to_string())
    save_output(yearly, "stage13_calendar_years")
    per = monthly.index.to_period("M")
    halves = []
    for label, mask in (("2008-04 to 2017-06", per < pd.Period("2017-07", "M")),
                        ("2017-07 to 2026-09", per >= pd.Period("2017-07", "M"))):
        for n in ["V2", "TREND_U2", "CARRY_U2", "COMBO_U2"]:
            x = monthly.loc[mask, n]
            halves.append({"half": label, "strategy": n, "sharpe": x.mean() / x.std() * np.sqrt(12),
                           "sharpe_se_lo": lo_sharpe_se(x), "ann_ret_pct": x.mean() * 1200})
    halves = pd.DataFrame(halves)
    print("\nHalves:")
    print(halves.pivot(index="strategy", columns="half", values=["sharpe", "ann_ret_pct"]).round(3).to_string())
    save_output(halves.set_index(["half", "strategy"]), "stage13_subperiods")

    # --- Crisis windows -------------------------------------------------------------
    spy_level = (1 + p.rets["SPY"].loc[p.dates[0]:p.end].iloc[1:]).cumprod()
    spy_level = pd.concat([pd.Series(1.0, index=[p.dates[0]]), spy_level])
    win = drawdown_windows(spy_level, CRISIS_DRAWDOWN_THRESHOLD)
    for n in ["V2", "TREND_U2", "CARRY_U2", "COMBO_U2", "60/40", "SPY"]:
        win[n] = [window_return(daily[n], r.peak, r.trough) * 100 for r in win.itertuples()]
    print("\nCrisis windows (cumulative excess return %, peak to trough):")
    print(win.drop(columns=["recovered"]).round(2).to_string(index=False))
    save_output(win.set_index("peak"), "stage13_crisis_windows")
    sm_rows = [{"strategy": n, **smile(quarterly[n], quarterly["SPY"])}
               for n in ["TREND_U2", "CARRY_U2", "COMBO_U2"]]
    print("\nSmile (quarterly, NW t):")
    print(pd.DataFrame(sm_rows).set_index("strategy").round(3).to_string())

    # --- Costs, financing, vol accuracy -----------------------------------------------
    cost_rows = []
    for n in ["TREND_U2", "CARRY_U2", "COMBO_U2", "COMBO_CORE"]:
        g_res, g_m = p.run(w[n], cost=0.0)
        g = summary(g_m, g_res.excess_returns)
        turn = g_res.turnover.iloc[1:].mean() * 12
        row = {"strategy": n, "turnover_yr": turn, "gross_sharpe": g["sharpe"],
               "breakeven_bps": g["ann_ret_pct"] / 100 / turn * 1e4}
        for c in COST_SENSITIVITY_BPS:
            if c:
                _, mc = p.run(w[n], cost=c)
                row[f"sharpe_{c}"] = mc.mean() / mc.std() * np.sqrt(12)
        _, mf = p.run(w[n], borrow_bps=100, financing_bps=100)
        row["sharpe_fees_100_100"] = mf.mean() / mf.std() * np.sqrt(12)
        cost_rows.append(row)
    costs = pd.DataFrame(cost_rows).set_index("strategy")
    print("\nCosts (Sharpe by one-way bps; break-even bps; with 100 bps borrow + 100 bps financing):")
    print(costs.round(3).to_string())
    save_output(costs, "stage13_costs")
    vb = {}
    for n in ["TREND_U2", "CARRY_U2", "COMBO_U2"]:
        ea = portfolio_ex_ante_vol(w[n], p.dex, p.vol)
        m = monthly[n]
        vb[n] = float((m.values / (ea.values / np.sqrt(12))).std(ddof=1))
    print("Vol bias statistic:", {k: round(v, 3) for k, v in vb.items()})
    cw = w["COMBO_U2"]
    print(f"COMBO_U2 gross leverage: mean {cw.abs().sum(axis=1).mean():.2f}, "
          f"p95 {cw.abs().sum(axis=1).quantile(0.95):.2f}, max {cw.abs().sum(axis=1).max():.2f}; "
          f"short notional mean {cw.clip(upper=0).abs().sum(axis=1).mean():.2f}")

    # --- 60/40 blends -------------------------------------------------------------------
    blend_rows = []
    for a in (0.0,) + tuple(BLEND_WEIGHTS):
        b = (1 - a) * monthly["60/40"] + a * monthly["COMBO_U2"]
        wealth = (1 + b).cumprod()
        blend_rows.append({"combo_weight": a, "ann_ret_pct": b.mean() * 1200,
                           "ann_vol_pct": b.std() * np.sqrt(12) * 100,
                           "sharpe": b.mean() / b.std() * np.sqrt(12),
                           "max_dd_monthly_pct": (wealth / wealth.cummax() - 1).min() * 100,
                           "ret_2022_pct": ((1 + b.loc["2022"]).prod() - 1) * 100})
    blends = pd.DataFrame(blend_rows).set_index("combo_weight")
    print("\n60/40 blended with COMBO_U2 (monthly rebalanced; COMBO net of costs):")
    print(blends.round(3).to_string())
    save_output(blends, "stage13_blends")


if __name__ == "__main__":
    main()
