"""LaTeX tables for the paper, built only from saved stage outputs.

Writes paper/tables/tab*.tex: booktabs `tabular` bodies only. Captions,
labels and notes live in main.tex, so wording can change without
regenerating numbers. Every number in the paper's prose should be traceable
to one of these files.
"""

import numpy as np
import pandas as pd

from config import MAIN_START, OUTPUT_DIR, PAPER_DIR
from data_io import load_frame
from inference import spanning_regression

TAB_DIR = PAPER_DIR / "tables"
NAME = {"V1": "V1 raw TSMOM", "V2a": "V2a inverse-vol", "V2": "V2 vol-targeted",
        "V3": "V3 regime (R1)", "V3_R2": "V3-R2 regime (SPY)", "V3_main": "V3 main-sample HMM",
        "V3_nodegen": "V3 excl.\\ degenerate fits", "V3_LA_smoothed": "Smoothed probabilities$^\\dagger$",
        "V3_LA_fullparams": "Full-sample parameters$^\\dagger$",
        "LO_RP": "Long-only risk parity", "SPY": "SPY", "60/40": "60/40", "V2_inst": "V2 per-instrument"}


SHORT = {"V3_R2": "V3-R2", "LO_RP": "LO-RP"}   # compact labels for the widest table


def csv(name, **kw):
    return pd.read_csv(OUTPUT_DIR / f"{name}.csv", **kw)


def num(x, d=2, sign=False, pct=False):
    """Format a number for LaTeX: true minus sign, fixed decimals."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if round(x, d) == 0:
        x = 0.0                      # no negative zero
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace("-", "$-$")
    return s + ("\\%" if pct else "")


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def tabular(cols: str, header: list, rows: list, groups: dict | None = None) -> str:
    """header: list of header lines (each a list of cells or a raw string)."""
    out = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
    for h in header:
        out.append(h if isinstance(h, str) else " & ".join(h) + " \\\\")
    out.append("\\midrule")
    for i, r in enumerate(rows):
        if groups and i in groups:
            if i:
                out.append("\\addlinespace")
            out.append(groups[i])
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def write(name, text):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / f"{name}.tex").write_text(text, encoding="ascii")
    print(f"wrote {name}")


# --------------------------------------------------------------------------
def tab_data():
    d = csv("data_summary", index_col=0)
    rows = [[t, d.loc[t, "class"], num(d.loc[t, "ann_excess_ret_pct"]), num(d.loc[t, "ann_vol_pct"]),
             num(d.loc[t, "ann_vol_daily_pct"]), num(d.loc[t, "sharpe"]),
             num(d.loc[t, "skew_monthly"]), num(d.loc[t, "worst_month_pct"], 1)] for t in d.index]
    write("tab01_data", tabular(
        "llrrrrrr",
        [["ETF", "Class", "Mean (\\%)", "Vol (\\%)", "Daily vol (\\%)", "Sharpe", "Skew", "Worst month (\\%)"]],
        rows))
    c = csv("monthly_correlations", index_col=0)
    rows = [[t] + [("" if j > i else num(c.loc[t, u])) for j, u in enumerate(c.columns)]
            for i, t in enumerate(c.index)]
    write("tab02_correlations", tabular("l" + "r" * len(c.columns), [[""] + list(c.columns)], rows))


def tab_predictive():
    p = csv("predictive_signals")
    rows = []
    for k in ["1", "3", "6", "12", "blend"]:
        a = p[(p.lookback.astype(str) == k) & (~p.fixed_effects)].iloc[0]
        b = p[(p.lookback.astype(str) == k) & (p.fixed_effects)].iloc[0]
        lab = f"{k}m" if k != "blend" else "Blend"
        rows.append([lab, num(a.coef, 3), num(a.t), num(b.coef, 3), num(b.t)])
    write("tab03_predictive", tabular(
        "lrrrr",
        ["& \\multicolumn{2}{c}{Pooled} & \\multicolumn{2}{c}{Instrument fixed effects} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         ["Signal $s^k_{t-1}$", "Coef.", "$t$", "Coef.", "$t$"]], rows))


def tab_performance():
    m = csv("stage6_core_metrics", index_col=0)
    order = ["V1", "V2a", "V2", "V3", "V3_R2", "LO_RP", "SPY", "60/40"]
    rows = []
    for n in order:
        r = m.loc[n]
        rows.append([SHORT.get(n, n), num(r.ann_ret_pct), num(r.ann_vol_pct), num(r.sharpe),
                     f"({num(r.sharpe_se_lo)})", num(r.nw_t), num(r.sortino), num(r.max_dd_pct, 1),
                     num(r.calmar), num(r.skew_d), num(r.skew_m), num(r.skew_q),
                     num(r.turnover_yr, 1), num(r.gross_lev_mean)])
    write("tab04_performance", tabular(
        "lrrrrrrrrrrrrr",
        [["", "Mean", "Vol", "Sharpe", "(SE)", "NW $t$", "Sortino", "MDD", "Calmar",
          "Skew$_d$", "Skew$_m$", "Skew$_q$", "Turn.", "Lev."]],
        rows, groups={0: "\\multicolumn{14}{l}{\\textit{Trend strategies}} \\\\",
                      5: "\\multicolumn{14}{l}{\\textit{Benchmarks}} \\\\"}))


def tab_sharpe_tests():
    a = csv("stage4_sharpe_tests")
    b = csv("stage5_sharpe_tests")
    rows = []
    for df in (a, b[b.a.isin(["V3", "V3_R2", "V3_main", "V3_nodegen"])]):
        for _, r in df.iterrows():
            rows.append([f"{NAME[r.a]} vs {NAME[r.b]}", num(r.sharpe_a), num(r.sharpe_b),
                         num(r["diff"], sign=True), f"[{num(r.ci_low)}, {num(r.ci_high)}]",
                         pval(r.p_value), num(r["corr"])])
    write("tab05_sharpe_tests", tabular(
        "lrrrcrr", [["Comparison", "SR$_a$", "SR$_b$", "Difference", "95\\% CI", "$p$", "Corr."]], rows))


def tab_spanning():
    s = csv("stage4_spanning")
    rows = []
    for y in ("V1", "V2"):
        for f in ("SPY", "60/40", "LO_RP", "SPY+LO_RP"):
            r = s[(s.y == y) & (s.factors == f)].iloc[0]
            b1 = r.get("beta_SPY") if f in ("SPY", "SPY+LO_RP") else r.get(f"beta_{f}")
            b2 = r.get("beta_LO_RP") if f == "SPY+LO_RP" else np.nan
            fname = {"LO_RP": "LO-RP", "SPY+LO_RP": "SPY + LO-RP"}.get(f, f)
            rows.append([y, fname, num(r.alpha_ann_pct), num(r.alpha_t), num(b1), num(b2), num(r.r2)])
    # External validation: V2 on the AQR TSMOM factor, aligned on calendar month.
    m = load_frame("stage6_monthly_returns", OUTPUT_DIR)
    aqr = load_frame("aqr_tsmom").loc[MAIN_START:]
    m.index, aqr.index = m.index.to_period("M"), aqr.index.to_period("M")
    common = m.index.intersection(aqr.index)
    assert len(common) == len(aqr)
    for y in ("V1", "V2"):
        r = spanning_regression(m.loc[common, y], aqr.loc[common, ["tsmom"]])
        rows.append([y, "AQR TSMOM", num(r["alpha_ann_pct"]), num(r["alpha_t"]), num(r["beta_tsmom"]),
                     "--", num(r["r2"])])
    write("tab06_spanning", tabular(
        "llrrrrr", [["Strategy", "Factors", "$\\alpha$ (\\%/yr)", "$t(\\alpha)$", "$\\beta_1$",
                     "$\\beta_2$", "$R^2$"]], rows))
    return {"aqr_months": len(common)}


def tab_crisis():
    w = csv("stage6_crisis_windows")
    rows = []
    for _, r in w.iterrows():
        rows.append([r.peak, r.trough, num(r.spy_dd * 100, 1), num(r.SPY, 1), num(r.V1, 1), num(r.V2, 1),
                     num(r.V3, 1), num(r.LO_RP, 1), num(r["60/40"], 1)])
    rows.append("\\midrule")
    q = csv("stage6_worst_decile")
    for freq, lab in (("month", "SPY worst-decile months, mean"), ("quarter", "SPY worst-decile quarters, mean")):
        x = q[q.freq == freq].set_index("strategy")
        n = int(x.n_worst.iloc[0])
        rows.append([f"\\multicolumn{{3}}{{l}}{{{lab} ($n={n}$)}}", num(x.loc["SPY", "mean_worst_pct"], 1),
                     num(x.loc["V1", "mean_worst_pct"], 1), num(x.loc["V2", "mean_worst_pct"], 1),
                     num(x.loc["V3", "mean_worst_pct"], 1), num(x.loc["LO_RP", "mean_worst_pct"], 1),
                     num(x.loc["60/40", "mean_worst_pct"], 1)])
    # The window is defined by SPY's total-return drawdown; every return column
    # (SPY included) is the excess return over T-bills across the same window.
    body = tabular("llrrrrrrr",
                   ["& & SPY & \\multicolumn{6}{c}{Excess return over the window (\\%)} \\\\",
                    "\\cmidrule(lr){4-9}",
                    ["Peak", "Trough", "drawdown", "SPY", "V1", "V2", "V3", "LO-RP", "60/40"]], rows)
    write("tab07_crisis", body)


def tab_regime():
    p = csv("stage5_performance", index_col=0)
    sd = csv("stage5_sharpe_tests", index_col=0)
    sp = csv("stage5_spanning", index_col=0)
    rows = []
    for n in ["V2", "V3", "V3_R2", "V3_main", "V3_nodegen", "V3_LA_smoothed", "V3_LA_fullparams"]:
        r = p.loc[n]
        d = sd.loc[n] if n in sd.index else None
        a = sp.loc[n] if n in sp.index else None
        rows.append([NAME[n], num(r.ann_ret_pct), num(r.ann_vol_pct), num(r.sharpe), num(r.max_dd_pct, 1),
                     num(r.mean_gross_lev),
                     num(d["diff"], sign=True) if d is not None else "--",
                     pval(d.p_value) if d is not None else "--",
                     num(a.alpha_ann_pct) if a is not None else "--",
                     num(a.alpha_t) if a is not None else "--"])
    write("tab08_regime", tabular(
        "lrrrrrrrrr",
        [["", "Mean", "Vol", "Sharpe", "MDD", "Lev.", "$\\Delta$SR", "$p$", "$\\alpha$ on V2", "$t(\\alpha)$"]],
        rows, groups={5: "\\multicolumn{10}{l}{\\textit{Lookahead illustrations (use future data; not results)}} \\\\"}))
    pr = csv("stage5_predictive", index_col=0)
    lab = {"R1": "R1 primary", "R2": "R2 (SPY)", "R1_main": "R1 main-sample", "R1_nodegen": "R1 excl.\\ degenerate",
           "LA_smoothed": "R1 smoothed$^\\dagger$"}
    rows = [[lab[k], num(r.slope_pct), num(r.t), num(r.sharpe_hi), str(int(r.n_hi)), num(r.sharpe_lo),
             str(int(r.n_lo))] for k, r in pr.iterrows()]
    write("tab09_regime_predictive", tabular(
        "lrrrrrr", ["& & & \\multicolumn{2}{c}{$p_t > 0.5$} & \\multicolumn{2}{c}{$p_t \\le 0.5$} \\\\",
                    "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                    ["Model", "Slope", "$t$", "Sharpe", "Months", "Sharpe", "Months"]], rows))


def tab_robustness():
    r = csv("stage6_robustness", index_col=0)
    label = {"primary (V2)": "Primary (V2)", "execution lag 1 day": "Execution lag 1 day",
             "EWMA com 20": "EWMA centre of mass 20 days", "EWMA com 120": "EWMA centre of mass 120 days",
             "equal risk per instrument": "Equal risk per instrument",
             "gross leverage cap 2x": "Gross leverage cap 2$\\times$",
             "gross leverage cap 3x": "Gross leverage cap 3$\\times$",
             "continuous t-stat signal": "Continuous $t$-statistic signal",
             "lookback 1m": "Lookback 1 month", "lookback 3m": "Lookback 3 months",
             "lookback 6m": "Lookback 6 months", "blend of 1/3/6/12m": "Blend of 1/3/6/12 months",
             "excluding Equity": "Excluding equity", "excluding Rates": "Excluding rates",
             "excluding Commodity": "Excluding commodities", "excluding FX": "Excluding FX"}
    rows = []
    for k, lab in label.items():
        x = r.loc[k]
        first = k == "primary (V2)"
        rows.append([lab, num(x.sharpe), "--" if first else num(x.diff_vs_V2, sign=True),
                     "--" if first else pval(x.p_vs_V2), num(x.corr_V2), num(x.ann_vol_pct, 1),
                     num(x.max_dd_pct, 1), num(x.turnover_yr, 1), num(x.vol_bias)])
    write("tab10_robustness", tabular(
        "lrrrrrrrr", [["Variant", "Sharpe", "$\\Delta$SR", "$p$", "Corr.", "Vol", "MDD", "Turn.", "Vol bias"]],
        rows, groups={1: "\\multicolumn{9}{l}{\\textit{Implementation and estimation}} \\\\",
                      7: "\\multicolumn{9}{l}{\\textit{Signal}} \\\\",
                      12: "\\multicolumn{9}{l}{\\textit{Leave one asset class out}} \\\\"}))
    h = csv("stage6_subperiods")
    halves = list(dict.fromkeys(h.half))
    rows = []
    for n in ["V1", "V2a", "V2", "V3", "V3_R2", "LO_RP", "SPY", "60/40"]:
        cells = [NAME[n]]
        for half in halves:
            x = h[(h.half == half) & (h.strategy == n)].iloc[0]
            cells += [num(x.ann_ret_pct), num(x.sharpe), f"({num(x.sharpe_se_lo)})"]
        rows.append(cells)
    write("tab11_subperiods", tabular(
        "lrrrrrr", [f"& \\multicolumn{{3}}{{c}}{{{halves[0]}}} & \\multicolumn{{3}}{{c}}{{{halves[1]}}} \\\\",
                    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                    ["", "Mean", "Sharpe", "(SE)", "Mean", "Sharpe", "(SE)"]], rows))


def tab_costs():
    c = csv("stage6_costs", index_col=0)
    levels = [0, 2, 5, 10, 20, 50]
    rows = [[NAME[n], num(c.loc[n, "turnover_yr"], 1)] + [num(c.loc[n, f"sharpe_{k}"]) for k in levels]
            + [num(c.loc[n, "breakeven_bps"], 0)] for n in ["V1", "V2a", "V2", "V3", "V3_R2", "LO_RP"]]
    write("tab12_costs", tabular(
        "lr" + "r" * len(levels) + "r",
        ["& & \\multicolumn{6}{c}{Sharpe ratio at one-way cost (bps)} & \\\\",
         "\\cmidrule(lr){3-8}",
         ["", "Turnover"] + [str(k) for k in levels] + ["Break-even (bps)"]], rows))
    f = csv("stage6_fees")
    rows = []
    for n in ["V2", "V3", "LO_RP"]:
        x = f[f.strategy == n]
        get = lambda fee, b: x[(x.fee == fee) & (x.bps == b)].iloc[0].sharpe  # noqa: E731
        rows.append([NAME[n], num(get("borrow", 0)), num(get("borrow", 50)), num(get("borrow", 100)),
                     num(get("financing", 50)), num(get("financing", 100)), num(get("both", 100))])
    write("tab13_fees", tabular(
        "lrrrrrr", ["& & \\multicolumn{2}{c}{Borrow fee (bps/yr)} & \\multicolumn{2}{c}{Financing spread (bps/yr)} & Both \\\\",
                    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
                    ["", "None", "50", "100", "50", "100", "100"]], rows))


def tab_risk():
    v = csv("stage4_vol_validation", index_col=0)
    rows = [[NAME[n], num(r.mean_ex_ante_pct), num(r.realised_pct), num(r.bias_stat),
             str(int(r.months_abs_z_gt_2)), str(int(r.months_abs_z_gt_3))] for n, r in v.iterrows()]
    write("tab14_vol_validation", tabular(
        "lrrrrr", [["", "Ex-ante vol (\\%)", "Realised vol (\\%)", "Bias statistic",
                    "Months $|z|>2$", "Months $|z|>3$"]], rows))
    rc = csv("stage4_risk_contributions", index_col=0)
    lev = csv("stage4_leverage", index_col=0)["gross"]
    rows = [[c] + [num(rc.loc[c, n] * 100, 1) for n in ("V1", "V2", "V2_inst")] for c in rc.index]
    write("tab15_risk_shares", tabular(
        "lrrr", [["Asset class", "V1", "V2", "V2 per-instrument"]], rows))
    stats = [("Mean", lev.mean()), ("Median", lev.median()), ("5th percentile", lev.quantile(0.05)),
             ("95th percentile", lev.quantile(0.95)), ("Maximum", lev.max())]
    rows = [[k, num(x)] for k, x in stats] + [
        ["Share of months above 2$\\times$", num((lev > 2).mean() * 100, 0, pct=True)],
        ["Share of months above 3$\\times$", num((lev > 3).mean() * 100, 0, pct=True)]]
    write("tab16_leverage", tabular("lr", [["V2 gross leverage at rebalance", "Value"]], rows))


def tab_dsr():
    d = csv("stage6_deflated_sharpe").iloc[0]
    e = csv("stage6_dsr_sensitivity")
    rows = [[str(int(r.n_trials)) + (" (all configurations)" if int(r.n_trials) == int(d.n_trials) else ""),
             num(r.sr0_ann), num(r.dsr)] for r in e.itertuples()]
    write("tab17_dsr", tabular("lrr", [["Number of trials $N$", "SR$_0$ (annual)", "DSR"]], rows))


def main():
    tab_dsr()
    tab_risk()
    tab_data()
    tab_predictive()
    tab_performance()
    tab_sharpe_tests()
    tab_spanning()
    tab_crisis()
    tab_regime()
    tab_robustness()
    tab_costs()


if __name__ == "__main__":
    main()
