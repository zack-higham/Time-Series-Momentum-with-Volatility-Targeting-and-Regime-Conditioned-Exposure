"""Stage 2: excess returns, ex-ante volatility, signals, predictive regressions.

Writes to output/:
  monthly_excess_returns.csv, month_end_vol.csv, signal_{k}.csv
  predictive_lags.csv          z_t on z_{t-h} and sign(r_{t-h}), h = 1..24
  predictive_signals.csv       z_t on s^k_{t-1}, k = 1,3,6,12, blend
  per_instrument_signal12.csv  instrument-level regressions on the 12m signal
  decomposition.csv            timing vs static parts of TSMOM returns
"""

import pandas as pd

from config import MAIN_START, PREDICTIVE_REG_MAX_LAG, PRIMARY_LOOKBACK, SAMPLE_END, newey_west_lags
from data_io import save_output
from predictive_regression import (decompose, lag_regressions, per_instrument_signal_regressions,
                                   scaled_returns, signal_regressions)
from returns import daily_excess_returns, load_prices, month_end_dates, monthly_excess_returns
from signals import all_signals, cumulative_excess_return
from volatility import ewma_vol


def main():
    prices = load_prices()
    ends = month_end_dates(prices.index)
    monthly = monthly_excess_returns(prices)
    vol = ewma_vol(daily_excess_returns(prices)).loc[ends]
    signals = all_signals(prices)
    z = scaled_returns(monthly, vol)
    months = monthly.loc[MAIN_START:SAMPLE_END].index
    print(f"Main sample: {len(months)} months, {months[0].date()} to {months[-1].date()}")

    save_output(monthly, "monthly_excess_returns")
    save_output(vol, "month_end_vol")
    for k, s in signals.items():
        save_output(s, f"signal_{k}")

    # Every regression observation must be complete in the main sample.
    assert z.loc[months].notna().all().all(), "missing scaled return in main sample"
    assert signals[PRIMARY_LOOKBACK].shift(1).loc[months].notna().all().all()

    lags = lag_regressions(z, months, PREDICTIVE_REG_MAX_LAG)
    save_output(lags.set_index("lag"), "predictive_lags")
    sig = signal_regressions(z, signals, months)
    save_output(sig.set_index("lookback"), "predictive_signals")

    nw = newey_west_lags(len(months))
    per_inst = per_instrument_signal_regressions(z, signals[PRIMARY_LOOKBACK], months, nw)
    save_output(per_inst, "per_instrument_signal12")

    # Decomposition for the sign strategy actually traded, and for the linear
    # strategy (position proportional to the vol-scaled 12m return) used in MOP.
    cum12 = cumulative_excess_return(prices, PRIMARY_LOOKBACK)
    linear_x = cum12 / vol
    decomp = pd.concat({"sign": decompose(z, signals[PRIMARY_LOOKBACK], months),
                        "linear": decompose(z, linear_x, months)})
    save_output(decomp, "decomposition")

    pd.set_option("display.width", 160)
    print("\nSignal regressions (coef in units of vol-scaled monthly return; t clustered by month):")
    print(sig.round(3).to_string(index=False))
    print("\nLag regressions, pooled, regressor = return (h: coef, t):")
    pooled = lags[(lags.regressor == "return") & ~lags.fixed_effects]
    fe = lags[(lags.regressor == "return") & lags.fixed_effects]
    sgn = lags[(lags.regressor == "sign") & ~lags.fixed_effects]
    print(pd.DataFrame({"pooled_t": pooled.set_index("lag")["t"], "fe_t": fe.set_index("lag")["t"],
                        "sign_t": sgn.set_index("lag")["t"]}).round(2).T.to_string())
    print(f"\nPer-instrument regressions on 12m signal (NW lags {nw}):")
    print(per_inst.round(3).to_string())
    print("\nDecomposition of E[x_(t-1) z_t] (sign and linear 12m strategies):")
    print(decomp.round(4).to_string())
    for name in ("sign", "linear"):
        d = decomp.loc[name]
        print(f"  {name}: average total {d['total'].mean():.4f} = timing {d['timing'].mean():.4f}"
              f" + static {d['static'].mean():.4f} (timing share {d['timing'].mean() / d['total'].mean():.0%})")


if __name__ == "__main__":
    main()
