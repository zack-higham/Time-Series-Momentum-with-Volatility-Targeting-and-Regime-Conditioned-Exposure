"""Data-quality checks for the ETF panel, the risk-free rate and the AQR factor.

Each check prints its result; anything needing a human verdict is listed
explicitly rather than silently passed. Summary tables used in the paper's
Data section are written to output/.
"""

import numpy as np
import pandas as pd

from config import ASSET_CLASSES, INSTRUMENTS, MAIN_START, SAMPLE_END
from data_io import load_frame, save_output
from riskfree import daily_rf

EXTREME_MOVE = 0.08      # flag daily |return| above 8%
ZERO_RUN_FLAG = 3        # flag runs of >= 3 consecutive zero returns


def section(title):
    print(f"\n=== {title} ===")


def check_coverage(adj, classes=INSTRUMENTS):
    section("1. Coverage and calendar")
    calendar = adj["SPY"].dropna().index
    assert adj.index.is_monotonic_increasing and not adj.index.has_duplicates
    assert adj.index[-1] == pd.Timestamp(SAMPLE_END), "sample must end on the last complete day"
    rows = []
    for t in adj.columns:
        s = adj[t]
        first, last = s.first_valid_index(), s.last_valid_index()
        inside = calendar[(calendar >= first) & (calendar <= last)]
        gaps = int(s.reindex(inside).isna().sum())
        nonpositive = int((s.dropna() <= 0).sum())
        rows.append({"ticker": t, "class": classes[t], "first": first.date(),
                     "last": last.date(), "days": int(s.notna().sum()),
                     "interior_gaps": gaps, "nonpositive_prices": nonpositive})
    table = pd.DataFrame(rows).set_index("ticker")
    print(table.to_string())
    assert table["interior_gaps"].sum() == 0, "missing prices inside a ticker's history"
    assert table["nonpositive_prices"].sum() == 0
    return table


def check_total_return(close, adj, actions):
    section("2. Total-return adjustment (rebuilt from close + distributions)")
    div = (actions.pivot_table(index=actions.index, columns="ticker", values="dividend",
                               aggfunc="sum")
           .reindex(index=close.index, columns=close.columns).fillna(0.0))
    # Two reinvestment conventions on an ex-date with distribution D:
    #   additive (exact one-day holding return): (C_t + D) / C_{t-1} - 1
    #   Yahoo's multiplicative adjustment:        C_t / (C_{t-1} - D) - 1
    # Yahoo's series must match its own convention exactly (verifies the
    # data); the gap to the additive return is a small convention bias,
    # largest for high-yield, high-volatility funds, and is reported.
    rows = []
    for t in close.columns:
        c = close[t].dropna()
        d = div[t].reindex(c.index)
        additive = (c + d) / c.shift(1) - 1
        multiplicative = c / (c.shift(1) - d) - 1
        yahoo = adj[t].dropna().pct_change()
        price_only = c.pct_change()
        yearly = lambda r: (1 + r).groupby(r.index.year).prod() - 1
        income = (yearly(yahoo) - yearly(price_only)).mean() * 100
        rows.append({"ticker": t, "distributions": int((d > 0).sum()),
                     "yahoo_conv_max_annual_gap_bps":
                         (yearly(multiplicative) - yearly(yahoo)).abs().max() * 1e4,
                     "additive_max_annual_gap_bps": (yearly(additive) - yearly(yahoo)).abs().max() * 1e4,
                     "additive_mean_annual_gap_bps": (yearly(yahoo) - yearly(additive)).mean() * 1e4,
                     "mean_income_pct_yr": income})
    table = pd.DataFrame(rows).set_index("ticker").round(2)
    print(table.to_string())
    assert (table["yahoo_conv_max_annual_gap_bps"] < 1).all(), \
        "adjusted series disagrees with close + distributions"
    splits = actions[actions["split"] > 0]
    print(f"Splits in the data: {len(splits)}")
    for date, row in splits.iterrows():
        c = close[row["ticker"]]
        jump = c.loc[date] / c.loc[:date].iloc[-2] - 1
        print(f"  {row['ticker']} {date.date()} ratio {row['split']:g}: close move that day {jump:+.2%}")
        assert abs(jump) < 0.10, "unadjusted split in the close series"


def check_stale_and_extreme(adj, volume):
    section("3. Stale prices and extreme moves")
    returns = adj.pct_change(fill_method=None)
    for t in adj.columns:
        r = returns[t].dropna()
        zero = (r == 0)
        run_id = (zero != zero.shift()).cumsum()
        runs = zero.groupby(run_id).sum()
        long_runs = runs[runs >= ZERO_RUN_FLAG]
        zero_vol = int((volume[t].reindex(r.index) == 0).sum())
        print(f"{t}: zero-return days {zero.mean():.1%}, longest run {int(runs.max())}, "
              f"runs >= {ZERO_RUN_FLAG}: {len(long_runs)}, zero-volume days {zero_vol}")
        for rid in long_runs.index:
            days = r.index[run_id == rid]
            print(f"    zero run {days[0].date()} to {days[-1].date()} ({len(days)} days)")

    big = returns.stack()
    big = big[big.abs() > EXTREME_MOVE].sort_index()
    print(f"\nDaily moves above {EXTREME_MOVE:.0%} in absolute value: {len(big)}")
    for (date, t), r in big.items():
        print(f"  {date.date()} {t} {r:+.1%}")


def check_risk_free(trading_days):
    section("4. Risk-free rate")
    irx = load_frame("irx")["irx_pct"]
    missing = trading_days.difference(irx.index)
    print(f"ETF trading days without an ^IRX quote (forward-filled): {len(missing)}")
    print("  " + ", ".join(d.strftime("%Y-%m-%d (%a)") for d in missing))
    print(f"^IRX range {irx.min():.3f}% to {irx.max():.3f}%; negative days: {(irx < 0).sum()}")

    rf = daily_rf(trading_days).dropna()
    french = load_frame("french_rf")["french_rf_pct_monthly"] / 100.0
    monthly = pd.DataFrame({
        "irx": (1 + rf).resample("ME").prod() - 1,
        "french": french,
    }).dropna()
    diff_bps_yr = (monthly["irx"] - monthly["french"]).mean() * 12 * 1e4
    # French is the one-month bill, ^IRX the three-month bill, so small gaps are
    # expected around rate changes; the check is for level and timing errors.
    print(f"Monthly rf, ^IRX-based vs French (overlap {monthly.index[0].date()} to "
          f"{monthly.index[-1].date()}): corr {monthly.corr().iloc[0, 1]:.4f}, "
          f"mean difference {diff_bps_yr:+.1f} bps/yr, "
          f"max abs monthly gap {(monthly['irx'] - monthly['french']).abs().max() * 1e4:.1f} bps")
    yearly = (1 + rf).groupby(rf.index.year).prod() - 1
    print("Annual rf from ^IRX (%): " + ", ".join(f"{y}: {v * 100:.2f}" for y, v in yearly.items()))


def summarise_main_sample(adj, trading_days):
    section("5. Main-sample summary statistics (excess returns)")
    rf = daily_rf(trading_days)
    returns = adj.pct_change(fill_method=None)
    excess = returns.sub(rf, axis=0).loc[f"{MAIN_START}-01":SAMPLE_END]
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly_rf = (1 + rf).resample("ME").prod() - 1
    monthly_ex = monthly.sub(monthly_rf, axis=0).loc[MAIN_START:SAMPLE_END]
    stats = pd.DataFrame({
        "class": pd.Series(INSTRUMENTS),
        "ann_excess_ret_pct": monthly_ex.mean() * 12 * 100,
        # Mean, volatility and Sharpe all from monthly returns, so Sharpe =
        # return / vol within the table. Daily-annualised vol is reported
        # separately: for equities it is higher (negative daily autocorrelation
        # in 2008-09), and mixing the two makes a table look inconsistent.
        "ann_vol_pct": monthly_ex.std() * np.sqrt(12) * 100,
        "ann_vol_daily_pct": excess.std() * np.sqrt(252) * 100,
        "sharpe": monthly_ex.mean() / monthly_ex.std() * np.sqrt(12),
        "skew_monthly": monthly_ex.skew(),
        "worst_month_pct": monthly_ex.min() * 100,
        "months": monthly_ex.count(),
    }).round(2)
    print(stats.to_string())
    assert (stats["months"] == 222).all(), "main sample should be 222 complete months"
    save_output(stats, "data_summary")

    corr = monthly_ex.corr().round(2)
    save_output(corr, "monthly_correlations")
    print("\nWithin-class effective number of bets (eigenvalue participation ratio):")
    for cls in ASSET_CLASSES:
        cols = [t for t, c in INSTRUMENTS.items() if c == cls]
        eig = np.linalg.eigvalsh(monthly_ex[cols].corr().values)
        print(f"  {cls}: {len(cols)} ETFs, {eig.sum() ** 2 / (eig ** 2).sum():.2f} effective bets")


def check_aqr():
    section("6. AQR TSMOM benchmark")
    aqr = load_frame("aqr_tsmom")
    main = aqr.loc[MAIN_START:]
    print(f"Months: {len(aqr)} ({aqr.index[0].date()} to {aqr.index[-1].date()}); "
          f"main-sample overlap {len(main)} months; missing values {int(aqr.isna().sum().sum())}")
    stats = pd.DataFrame({"ann_ret_pct": main.mean() * 1200,
                          "ann_vol_pct": main.std() * np.sqrt(12) * 100,
                          "sharpe": main.mean() / main.std() * np.sqrt(12)}).round(2)
    print(stats.to_string())


def main():
    adj = load_frame("adj_close").loc[:SAMPLE_END]
    close = load_frame("close").loc[:SAMPLE_END]
    volume = load_frame("volume").loc[:SAMPLE_END]
    actions = load_frame("actions")
    trading_days = adj.index

    check_coverage(adj)
    check_total_return(close, adj, actions)
    check_stale_and_extreme(adj, volume)
    check_risk_free(trading_days)
    summarise_main_sample(adj, trading_days)
    check_aqr()


if __name__ == "__main__":
    main()
