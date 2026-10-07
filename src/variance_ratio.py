"""Daily versus monthly volatility of the equity funds (pre-registered in config.py, 2026-10-07).

VR = Var(monthly excess return) / (21 x Var(daily excess return)) for SPY, EFA
and EEM over three windows, with exactly Table 1's return definitions
(check_data.summarise_main_sample). VR < 1 means monthly risk is below what
daily volatility implies, the signature of negatively autocorrelated daily
returns. Writes output/variance_ratio.csv.
"""

import numpy as np
import pandas as pd

from config import MAIN_START, OUTPUT_DIR, SAMPLE_END, VR_CLOSE_TO_ONE, VR_FUNDS, VR_WINDOWS
from data_io import load_frame, save_output
from returns import load_prices
from riskfree import daily_rf


def main():
    adj = load_prices()
    rf = daily_rf(adj.index)
    returns = adj.pct_change(fill_method=None)
    daily_ex = returns.sub(rf, axis=0)
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly_rf = (1 + rf).resample("ME").prod() - 1
    monthly_ex = monthly.sub(monthly_rf, axis=0)

    # Reproduce Table 1's daily and monthly volatilities before using these series.
    table1 = load_frame("data_summary", OUTPUT_DIR)
    d = daily_ex.loc[f"{MAIN_START}-01":SAMPLE_END]
    m = monthly_ex.loc[MAIN_START:SAMPLE_END]
    assert len(m) == 222
    for f in VR_FUNDS:
        assert round(d[f].std() * np.sqrt(252) * 100, 2) == table1.loc[f, "ann_vol_daily_pct"], f
        assert round(m[f].std() * np.sqrt(12) * 100, 2) == table1.loc[f, "ann_vol_pct"], f
    print("Daily and monthly volatilities reproduce Table 1.")

    rows = []
    for window, (start, end) in VR_WINDOWS.items():
        dw = daily_ex.loc[f"{start}-01":pd.Period(end).end_time.normalize()]
        mw = monthly_ex.loc[start:end]
        for f in VR_FUNDS:
            vr = mw[f].var() / (21 * dw[f].var())
            # first-order autocorrelation of daily returns in the window, for interpretation
            rows.append({"window": window, "fund": f, "months": mw[f].count(), "days": dw[f].count(),
                         "vol_monthly_pct": mw[f].std() * np.sqrt(12) * 100,
                         "vol_daily_pct": dw[f].std() * np.sqrt(252) * 100,
                         "vr": vr, "daily_autocorr_1": dw[f].autocorr(1)})
    out = pd.DataFrame(rows).set_index(["window", "fund"])
    print(out.round(3).to_string())
    save_output(out, "variance_ratio")

    # Interpretation rule (fixed in config.py before this was run)
    vr = out["vr"].unstack("window")
    full, crisis, post = vr["Full sample"], vr["2008-04 to 2009-12"], vr["2010-01 to 2026-09"]
    c1, c2, c3 = full < 1, crisis < full, post >= VR_CLOSE_TO_ONE
    print("\n(i) full VR < 1:", c1.to_dict())
    print("(ii) 2008-09 VR < full VR:", c2.to_dict())
    print(f"(iii) 2010-26 VR >= {VR_CLOSE_TO_ONE}:", c3.to_dict())
    if (c1 & c2 & c3).all():
        verdict = "supported"
    elif (c1 & c2).all():
        verdict = "partially supported: the gap is not confined to 2008-09"
    else:
        verdict = "not supported: correct the caption"
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()
