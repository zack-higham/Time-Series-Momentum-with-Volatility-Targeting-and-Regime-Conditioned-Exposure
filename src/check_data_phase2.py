"""Phase 2 data checks: universe rule, U2 panel quality, rates, historical data.

Same standard as check_data.py. Anything needing a human verdict is printed
rather than silently passed. Report: output/data_checks_phase2.txt (via redirect).
"""

import numpy as np
import pandas as pd

from check_data import check_coverage, check_stale_and_extreme, check_total_return, section
from config import (DATA_DIR, FX_RATE_SERIES, GBP_SPLICE_SERIES, HIST_BILL, HIST_BOND_YIELDS, HIST_END,
                    HIST_FX, HIST_FX_RATES, HIST_START, INSTRUMENTS, MAIN_START,
                    PHASE2_ADDITIONS, PHASE2_FIRST_TRADE_BY, PHASE2_MIN_DOLLAR_VOLUME_M,
                    SAMPLE_END, UNIVERSE_U2)
from data_io import load_frame, save_output
from returns import daily_excess_returns, load_prices_u2, monthly_excess_returns
from signals import tsmom_signal
from volatility import ewma_vol

# Funds failing a rule that a mechanical screen cannot evaluate from
# first-trade date and volume alone (rules 3 to 5 in config).
QUALITATIVE_EXCLUSIONS = {"UDN": "inverse (rule 3)", "IAU": "same underlying as GLD (rule 4)",
                          "SHY": "phase 1 exclusion, ~1.5% volatility (rule 5)"}


def check_universe_rule():
    section("1. Universe rule re-applied from saved screening data")
    s = pd.read_csv(DATA_DIR / "phase2_screen.csv", index_col=0)
    s["first_trade"] = pd.to_datetime(s["first_trade"])
    ok_date = s["first_trade"] <= pd.Timestamp(PHASE2_FIRST_TRADE_BY)
    ok_liq = s["median_dollar_volume_m"] >= PHASE2_MIN_DOLLAR_VOLUME_M
    s["passes"] = ok_date & ok_liq & ~s.index.isin(list(QUALITATIVE_EXCLUSIONS))
    s["reason"] = np.where(~ok_date, "first trade after cutoff or no data",
                           np.where(~ok_liq, "below liquidity floor",
                                    s.index.map(lambda t: QUALITATIVE_EXCLUSIONS.get(t, ""))))
    print(s.to_string())
    selected = set(s.index[s["passes"]])
    assert selected == set(PHASE2_ADDITIONS), (selected ^ set(PHASE2_ADDITIONS))
    print(f"Rule selects exactly the {len(selected)} pre-registered additions; "
          f"U2 = {len(UNIVERSE_U2)} ETFs.")


def check_signals_available(adj):
    section("2. Every U2 fund has a 12m signal and a volatility estimate at the first rebalance")
    sig = tsmom_signal(adj, 12)
    vol = ewma_vol(daily_excess_returns(adj))
    first = pd.Timestamp("2008-03-31")
    missing_sig = sig.loc[first].isna()
    missing_vol = vol.loc[first].isna()
    print(f"Signals missing at {first.date()}: {list(missing_sig[missing_sig].index)}; "
          f"vols missing: {list(missing_vol[missing_vol].index)}")
    assert not missing_sig.any() and not missing_vol.any()
    m = monthly_excess_returns(adj).loc[MAIN_START:SAMPLE_END]
    assert len(m) == 222 and m.notna().all().all(), "every U2 fund needs 222 main-sample months"
    print("All 34 funds have 222 complete main-sample months.")
    return m


def check_rates():
    section("3. Short rates for FX carry")
    r = load_frame("fred_monthly_rates")
    for ccy, sid in FX_RATE_SERIES.items():
        x = r[sid].loc["2006":]
        print(f"  {ccy} {sid}: {x.first_valid_index().date()} to {x.last_valid_index().date()}, "
              f"gaps {int(x.loc[:x.last_valid_index()].isna().sum())}, "
              f"range {x.min():.2f}% to {x.max():.2f}%")
    d = load_frame("fred_daily")[GBP_SPLICE_SERIES]
    gbp3m = r[FX_RATE_SERIES["GBP"]]
    sonia_m = d.resample("MS").mean()
    overlap = pd.concat([gbp3m, sonia_m], axis=1).dropna().loc["2024":]
    print("  GBP 3m vs SONIA monthly mean, last 6 overlapping months (%):")
    print(overlap.tail(6).round(3).to_string())
    print(f"  Mean gap 2024 onwards {(overlap.iloc[:, 0] - overlap.iloc[:, 1]).mean():+.3f} pp: "
          f"the splice changes the sterling rate level by about this much from 2026-02.")


def check_historical():
    section("4. Historical validation inputs")
    d = load_frame("fred_daily")
    r = load_frame("fred_monthly_rates")
    f = load_frame("french_daily")
    lo, hi = f"{int(HIST_START[:4]) - 2}-01-01", "2008-03-31"
    for name, (sid, _) in {**HIST_FX, **HIST_BOND_YIELDS}.items():
        x = d[sid].loc[lo:hi]
        print(f"  {name} {sid}: obs {x.notna().sum()}, missing business days {int(x.isna().sum())}, "
              f"range {x.min():.3f} to {x.max():.3f}")
    x = d[HIST_BILL].loc[lo:hi]
    print(f"  bill {HIST_BILL}: obs {x.notna().sum()}, range {x.min():.2f}% to {x.max():.2f}%")
    for ccy, sid in HIST_FX_RATES.items():
        x = r[sid].loc[lo:hi]
        print(f"  rate {ccy} {sid}: {x.first_valid_index().date()} to {x.last_valid_index().date()}, "
              f"gaps {int(x.isna().sum())}")
        assert x.notna().all(), f"gap in {sid} over the historical window"
    print(f"  French daily market: {f.loc[lo:hi, 'mkt_rf'].notna().sum()} days")
    # FX spot moves above 5% in a day are rare: list them for a human check.
    for name, (sid, _) in HIST_FX.items():
        x = d[sid].loc[lo:hi].dropna()
        big = x.pct_change().abs()
        for day, v in big[big > 0.05].items():
            print(f"    {name} {day.date()} {v:+.1%}")


def main():
    check_universe_rule()
    adj = load_prices_u2()
    close = pd.concat([load_frame("close"), load_frame("u2add_close")], axis=1)[list(UNIVERSE_U2)]
    volume = pd.concat([load_frame("volume"), load_frame("u2add_volume")], axis=1)[list(UNIVERSE_U2)]
    actions = pd.concat([load_frame("actions"), load_frame("u2add_actions")]).sort_index()
    add = list(PHASE2_ADDITIONS)
    check_coverage(adj, classes=UNIVERSE_U2)
    check_total_return(close[add], adj[add], actions[actions["ticker"].isin(add)])
    check_stale_and_extreme(adj[add].loc[:SAMPLE_END], volume[add])
    m = check_signals_available(adj)
    section("5. Main-sample summary of the additions (monthly excess returns)")
    stats = pd.DataFrame({
        "class": pd.Series(UNIVERSE_U2),
        "ann_excess_ret_pct": m.mean() * 1200,
        "ann_vol_pct": m.std() * np.sqrt(12) * 100,
        "sharpe": m.mean() / m.std() * np.sqrt(12),
    }).round(2)
    print(stats.loc[add].to_string())
    save_output(stats, "data_summary_u2")
    save_output(m.corr().round(2), "monthly_correlations_u2")
    check_rates()
    check_historical()


if __name__ == "__main__":
    main()
