"""Figures for the paper, built only from saved stage outputs (output/*.csv).

Writes paper/figures/fig*.pdf (vector, for LaTeX). Pass --png DIR to also
write PNG previews for visual checking.

Style: one fixed colour per strategy in every figure (colour follows the
entity), benchmarks in grey, hairline grids, legends on every multi-series
plot plus direct end labels where lines are long.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import OUTPUT_DIR, PAPER_DIR, PORTFOLIO_VOL_TARGET  # noqa: E402
from data_io import load_frame  # noqa: E402

FIG_DIR = PAPER_DIR / "figures"
WIDTH = 6.3        # inches: LaTeX text width with 1in margins on A4/letter

# Colour follows the entity. Validated palette (light surface, all-pairs CVD-safe
# for the four strategy hues); benchmarks in grey ink.
COLOR = {"V2": "#2a78d6", "V1": "#eb6834", "V3": "#1baf7a", "LO_RP": "#4a3aa7",
         "V3_R2": "#e87ba4", "SPY": "#52514e", "60/40": "#898781"}
STYLE = {"60/40": (0, (4, 2))}            # dashed only to separate the two greys
LABEL = {"V1": "V1 raw TSMOM", "V2": "V2 vol-targeted", "V3": "V3 regime overlay",
         "V3_R2": "V3-R2 SPY regime", "LO_RP": "Long-only risk parity", "SPY": "SPY",
         "60/40": "60/40"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SHADE = "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.dpi": 150,
})


def out(name, dates=True):
    if dates:
        return load_frame(name, OUTPUT_DIR)
    return pd.read_csv(OUTPUT_DIR / f"{name}.csv", index_col=0)


def wealth(daily: pd.Series) -> pd.Series:
    w = (1 + daily.dropna()).cumprod()
    start = w.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series(1.0, index=[start]), w])


def end_labels(ax, series: dict, fmt="{:.2f}", min_gap=None):
    """Direct labels at the right end of each line, nudged apart vertically."""
    items = sorted(((s.iloc[-1], k) for k, s in series.items()), reverse=False)
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    tr = (np.log10 if log else (lambda v: v))
    gap = min_gap if min_gap is not None else 0.06 * (tr(hi) - tr(lo))
    placed = []
    for v, k in items:
        y = tr(v)
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        yy = 10 ** y if log else y
        ax.annotate(f"{LABEL[k].split(' ')[0]} {fmt.format(v)}", xy=(1.0, yy),
                    xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK2)


def plot_lines(ax, frame: pd.DataFrame, names):
    for n in names:
        ax.plot(frame.index, frame[n], color=COLOR[n], lw=1.6 if n == "V2" else 1.1,
                ls=STYLE.get(n, "-"), label=LABEL[n], zorder=3 if n == "V2" else 2)


def fig_cumulative(save):
    d = out("stage6_daily_returns")
    names = ["V2", "V3", "V1", "LO_RP", "SPY", "60/40"]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.6), sharex=True)
    for ax, scaled in zip(axes, (False, True)):
        series = {}
        for n in names:
            x = d[n].dropna()
            if scaled:   # ex post scaling to 10% realised vol: comparison only, not tradable
                x = x * PORTFOLIO_VOL_TARGET / (x.std() * np.sqrt(252))
            series[n] = wealth(x)
        frame = pd.DataFrame(series)
        plot_lines(ax, frame, names)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("Growth of 1 (excess of T-bills, log)")
        ax.set_title("(a) As traded" if not scaled else
                     "(b) Each series rescaled ex post to 10% realised volatility")
        end_labels(ax, {n: frame[n] for n in names})
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set_xlim(d.index[0] - pd.Timedelta(days=30), d.index[-1] + pd.Timedelta(days=30))
    save(fig, "fig01_cumulative")


def fig_drawdowns(save):
    d = out("stage6_daily_returns")
    names = ["V2", "V1", "LO_RP", "SPY"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    dd = pd.DataFrame({n: (lambda w: w / w.cummax() - 1)(wealth(d[n])) * 100 for n in names})
    plot_lines(ax, dd, names)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(-55, 2)
    save(fig, "fig02_drawdowns")


def fig_crisis(save):
    d = out("stage6_daily_returns")
    win = out("stage6_crisis_windows")
    names = ["V2", "V3", "V1", "SPY"]
    picks = ["2008-05-19", "2020-02-19", "2022-01-03"]
    titles = ["(a) Global financial crisis", "(b) COVID-19 crash", "(c) 2022 inflation shock"]
    mdates = matplotlib.dates
    ticks = [(mdates.MonthLocator(bymonth=(9, 1)), "%b %y"),
             (mdates.WeekdayLocator(byweekday=mdates.MO, interval=2), "%d %b"),
             (mdates.MonthLocator(bymonth=(1, 5, 9)), "%b %y")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=False)
    for ax, peak, title, (loc, fmt) in zip(axes, picks, titles, ticks):
        trough = pd.Timestamp(win.loc[peak, "trough"])
        x = d.loc[peak:trough].iloc[1:]
        cum = pd.DataFrame({n: wealth(x[n]) * 100 - 100 for n in names})
        cum.index = pd.DatetimeIndex([pd.Timestamp(peak)]).append(cum.index[1:])   # start at the peak
        plot_lines(ax, cum, names)
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax.set_title(title)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.tick_params(axis="x", labelsize=7)
        for n in ("V2", "SPY"):
            ax.annotate(f"{cum[n].iloc[-1]:+.1f}%", xy=(cum.index[-1], cum[n].iloc[-1]),
                        xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2,
                        va="center")
    axes[0].set_ylabel("Cumulative excess return (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig03_crisis_zooms")


def fig_regimes(save):
    probs = out("stage5_probabilities")
    wf = out("stage5_walkforward_R1")
    tp = out("stage5_weekly_trend_payoff")
    # Each probability is set at month-end t and applies to month t+1: plot as
    # a step held over the following month.
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9]})
    deg = wf.index[wf["degenerate"].astype(bool)]
    # Shade the holding month of each degenerate refit (they are not contiguous).
    # A probability set at month-end t (last trading day) is held through the
    # end of calendar month t+1.
    for d in deg:
        end = (d.to_period("M") + 1).to_timestamp(how="end")
        axes[0].axvspan(d, end, color=SHADE, zorder=0, lw=0)
    ax = axes[0]
    ax.fill_between([], [], color=SHADE, label=f"Degenerate fit ({len(deg)} months)")
    ax.step(probs.index, probs["R1"], where="post", color=COLOR["V3"], lw=1.1,
            label="Primary (trained from 2003)")
    ax.step(probs.index, probs["R1_main"], where="post", color=COLOR["V2"], lw=0.9,
            label="Main-sample only (live 2011-03)")
    ax.set_ylabel("P(trending)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) R1: probability of the high-mean state of the trend payoff (filtered)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=7,
              handlelength=1.5, columnspacing=1.2)
    ax = axes[1]
    ax.step(probs.index, probs["R2"], where="post", color=COLOR["V3_R2"], lw=1.1)
    ax.set_ylabel("P(turbulent)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) R2: probability of the high-variance state of SPY (filtered)")
    ax = axes[2]
    tpm = tp.loc[probs.index[0] - pd.Timedelta(days=7):, "y"]
    rv = tpm.rolling(13).std() * np.sqrt(52)
    ax.plot(rv.index, rv, color=INK2, lw=0.9)
    ax.set_ylabel("Annualised")
    ax.set_title("(c) Realised volatility of the weekly trend payoff (13-week rolling)")
    ax.set_xlim(probs.index[0], probs.index[-1] + pd.offsets.MonthEnd(1))
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig04_regimes")


def fig_predictive(save):
    p = out("predictive_lags", dates=False).reset_index()
    p = p[~p["fixed_effects"].astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, reg, title in zip(axes, ("return", "sign"),
                              ("(a) Regressor: scaled return at lag h",
                               "(b) Regressor: sign of return at lag h")):
        q = p[p["regressor"] == reg].sort_values("lag")
        ax.bar(q["lag"], q["t"], width=0.7, color=COLOR["V2"], zorder=2)
        for y in (1.96, -1.96):
            ax.axhline(y, color=MUTED, lw=0.6, zorder=1)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Lag h (months)")
        ax.set_xticks([1, 6, 12, 18, 24])
    axes[0].set_ylabel("t-statistic (clustered by month)")
    axes[1].annotate("$\\pm$1.96", xy=(24.6, 1.96), fontsize=7, color=MUTED, va="bottom", ha="right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig05_predictive_tstats")


def fig_horizons(save):
    r = out("stage6_robustness", dates=False)
    rows = [("lookback 1m", "1m"), ("lookback 3m", "3m"), ("lookback 6m", "6m"),
            ("primary (V2)", "12m\n(primary)"), ("blend of 1/3/6/12m", "Blend"),
            ("continuous t-stat signal", "12m\ncontinuous")]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.6))
    for i, (key, lab) in enumerate(rows):
        sr, se = r.loc[key, "sharpe"], r.loc[key, "sharpe_se_lo"]
        c = COLOR["V2"] if key == "primary (V2)" else INK2
        ax.plot([i, i], [sr - 1.96 * se, sr + 1.96 * se], color=c, lw=1.2, zorder=2)
        ax.plot(i, sr, "o", ms=5, color=c, mec="white", mew=1.0, zorder=3)
        ax.annotate(f"{sr:.2f}", xy=(i, sr), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.set_xticks(range(len(rows)), [lab for _, lab in rows])
    ax.set_ylabel("Sharpe ratio (net, annualised)")
    ax.set_xlim(-0.5, len(rows) - 0.3)
    ax.grid(axis="x", visible=False)
    save(fig, "fig06_horizons")


def fig_smile(save):
    q = out("stage6_quarterly_returns", dates=False) * 100
    sm = out("stage6_smile", dates=False)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 3.0))
    ax.scatter(q["SPY"], q["V2"], s=14, color=COLOR["V2"], edgecolor="white", lw=0.6, zorder=3)
    xs = np.linspace(q["SPY"].min(), q["SPY"].max(), 200)
    a, b, c = sm.loc["V2", "a_pct"], sm.loc["V2", "b"], sm.loc["V2", "c"]
    ax.plot(xs, a + b * xs + c * xs ** 2 / 100, color=INK, lw=1.0, zorder=2)
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.axvline(0, color=AXIS, lw=0.6)
    ax.set_xlabel("SPY quarterly excess return (%)")
    ax.set_ylabel("V2 quarterly excess return (%)")
    ax.annotate(f"quadratic coefficient {c:.2f} (t = {sm.loc['V2', 't_c']:.2f})",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7, color=INK2)
    save(fig, "fig07_smile")


def fig_vol_leverage(save):
    roll = out("stage4_rolling_vol")
    lev = out("stage4_leverage")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.8), sharex=True)
    ax = axes[0]
    ax.plot(roll.index, roll["V2"], color=COLOR["V2"], lw=1.2, label="V2 realised (12-month rolling)")
    ax.axhline(PORTFOLIO_VOL_TARGET * 100, color=INK, lw=0.8, label="10% ex-ante target")
    ax.set_ylabel("Volatility (%)")
    ax.set_ylim(0, 19)
    ax.set_title("(a) Realised volatility against target")
    ax.legend(loc="lower left", ncol=2)
    ax = axes[1]
    ax.step(lev.index, lev["gross"], where="post", color=COLOR["V2"], lw=1.0)
    for y in (2, 3):
        ax.axhline(y, color=MUTED, lw=0.6)
        ax.annotate(f"{y}x", xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color=MUTED, va="center")
    ax.set_ylabel("Gross leverage (x NAV)")
    ax.set_title("(b) V2 gross leverage at each rebalance")
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig08_vol_leverage")


def fig_costs(save):
    c = out("stage6_costs", dates=False)
    levels = [0, 2, 5, 10, 20, 50]
    names = ["V2", "V3", "V1", "LO_RP"]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.7))
    for n in names:
        ys = [c.loc[n, f"sharpe_{k}"] for k in levels]
        ax.plot(levels, ys, color=COLOR[n], lw=1.6 if n == "V2" else 1.1, marker="o", ms=3.5,
                mec="white", mew=0.6, label=LABEL[n])
    ax.axvline(10, color=MUTED, lw=0.6)
    ax.annotate("assumed\n10 bps", xy=(10.8, 0.52), fontsize=7, color=MUTED, va="top")
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.set_xlabel("One-way cost per unit of notional traded (bps)")
    ax.set_ylabel("Sharpe ratio (net)")
    ax.legend(loc="upper right")
    save(fig, "fig09_costs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None)
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(FIG_DIR / f"{name}.pdf")
        if args.png:
            args.png.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.png / f"{name}.png", dpi=170)
        plt.close(fig)
        print(f"wrote {name}")

    for f in (fig_cumulative, fig_drawdowns, fig_crisis, fig_regimes, fig_predictive,
              fig_horizons, fig_smile, fig_vol_leverage, fig_costs):
        f(save)


if __name__ == "__main__":
    main()
