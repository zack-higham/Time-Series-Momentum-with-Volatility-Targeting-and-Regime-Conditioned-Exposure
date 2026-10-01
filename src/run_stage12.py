"""Stage 12: carry signals for the U2 universe.

Writes output/stage12_*.csv; report via redirect to output/stage12_report.txt.
  carry levels and summary by fund
  pooled predictive regressions of the volatility-scaled return on the lagged
    carry signal (rank and sign), all carry classes and by class (as Stage 2)
  average cross-sectional correlation of the carry and 12m trend signals
  diagnostic: carry with EWT's December 2022 and 2023 distributions removed
    (large capital-gains distributions that the pre-registered income
    measure cannot separate from dividends; the primary signal keeps them)
"""

import numpy as np
import pandas as pd

from carry import carry_panel, rank_signal, sign_signal
from config import CARRY_CLASSES, MAIN_START, SAMPLE_END, UNIVERSE_U2
from data_io import load_frame, save_output
from predictive_regression import _pooled_ols, scaled_returns
from returns import daily_excess_returns, load_prices_u2, month_end_dates, monthly_excess_returns
from signals import tsmom_signal
from volatility import ewma_vol

pd.set_option("display.width", 220)
EWT_CG_DATES = ["2022-12-13", "2023-12-20"]


def inputs():
    prices = load_prices_u2()
    close = pd.concat([load_frame("close"), load_frame("u2add_close")], axis=1)[list(UNIVERSE_U2)]
    actions = pd.concat([load_frame("actions"), load_frame("u2add_actions")]).sort_index()
    return prices, close, actions


def main():
    prices, close, actions = inputs()
    carry = carry_panel(UNIVERSE_U2, close, actions, prices.index)
    save_output(carry, "stage12_carry")
    funds = list(carry.columns)
    main_dates = carry.loc[pd.Timestamp("2008-03-31"):pd.Timestamp("2026-08-31")].index
    assert len(main_dates) == 222 and carry.loc[main_dates].notna().all().all()

    c = carry.loc[main_dates] * 100
    summ = pd.DataFrame({"class": pd.Series(UNIVERSE_U2)[funds], "mean_pct": c.mean(),
                         "min_pct": c.min(), "max_pct": c.max(), "share_positive": (c > 0).mean()})
    print("Carry at rebalance dates 2008-03 to 2026-08 (annual %, in excess of cash):")
    print(summ.round(2).to_string())
    save_output(summ, "stage12_carry_summary")

    # --- Predictive regressions (z_t on signal at t-1), as Stage 2 -------------
    monthly = monthly_excess_returns(prices[funds])
    ends = month_end_dates(prices.index)
    vol = ewma_vol(daily_excess_returns(prices[funds])).loc[ends]
    z = scaled_returns(monthly, vol)
    months = monthly.loc[MAIN_START:SAMPLE_END].index
    assert z.loc[months].notna().all().all()
    signals = {"rank": rank_signal(carry, UNIVERSE_U2), "sign": sign_signal(carry)}
    rows = []
    for name, s in signals.items():
        lagged = s.reindex(z.index).shift(1)
        for cls in ("All",) + CARRY_CLASSES:
            cols = funds if cls == "All" else [t for t in funds if UNIVERSE_U2[t] == cls]
            for fe in (False, True):
                rows.append({"signal": name, "class": cls, "fixed_effects": fe,
                             **_pooled_ols(z[cols], lagged[cols], months, fe)})
    pred = pd.DataFrame(rows)
    print("\nPooled regressions of the scaled return on the lagged carry signal (clustered by month):")
    print(pred.round(3).to_string(index=False))
    save_output(pred.set_index(["signal", "class", "fixed_effects"]), "stage12_predictive")

    # --- How different is carry from trend? --------------------------------------
    trend = tsmom_signal(prices, 12)[funds].loc[main_dates]
    rank = signals["rank"].loc[main_dates]
    corr = {}
    for cls in CARRY_CLASSES:
        cols = [t for t in funds if UNIVERSE_U2[t] == cls]
        per_date = [np.corrcoef(rank.loc[d, cols], trend.loc[d, cols])[0, 1]
                    for d in main_dates if trend.loc[d, cols].std() > 0]
        corr[cls] = np.nanmean(per_date)
    agree = (np.sign(rank) == trend).mean().mean()
    print("\nAverage cross-sectional correlation, carry rank vs 12m trend signal:",
          {k: round(float(v), 3) for k, v in corr.items()},
          f"| same direction in {agree:.0%} of fund-months")
    save_output(pd.Series(corr, name="corr"), "stage12_trend_carry_signal_corr")

    # --- Diagnostic: EWT capital-gains distributions removed ------------------------
    drop = actions.index.isin(pd.to_datetime(EWT_CG_DATES)) & (actions["ticker"] == "EWT")
    print(f"\nDiagnostic: removing {int(drop.sum())} EWT distributions "
          f"({', '.join(EWT_CG_DATES)}): "
          f"{actions.loc[drop, 'dividend'].round(3).tolist()} per share")
    carry_diag = carry_panel(UNIVERSE_U2, close, actions[~drop], prices.index)
    changed = (carry_diag - carry).abs().loc[main_dates] > 1e-12
    print(f"  carry changes in {int(changed.sum().sum())} fund-months, all EWT: "
          f"{set(changed.columns[changed.any()])}")
    rank_diag = rank_signal(carry_diag, UNIVERSE_U2).loc[main_dates]
    moved = (rank_diag - rank).abs().max(axis=1)
    print(f"  equity ranks change in {int((moved > 0).sum())} months; "
          f"EWT signal mean {rank['EWT'].loc['2022-12':'2024-11'].mean():.2f} -> "
          f"{rank_diag['EWT'].loc['2022-12':'2024-11'].mean():.2f} over 2022-12 to 2024-11")
    save_output(carry_diag, "stage12_carry_diag_ewt")


if __name__ == "__main__":
    main()
