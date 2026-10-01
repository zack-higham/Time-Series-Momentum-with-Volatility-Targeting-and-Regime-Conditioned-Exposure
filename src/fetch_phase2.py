"""Phase 2 downloads: the U2 additions, the universe screen, rates and history.

1. Every phase 2 candidate ETF (including those the rule excludes): first
   trade date and median daily dollar volume over the pre-sample liquidity
   window, saved as `phase2_screen.csv` so the universe rule can be re-applied
   from saved data. No returns are used.
2. Prices, volume and distributions for the 24 U2 additions, in the same
   format as phase 1. The 10 core ETFs are NOT re-downloaded: phase 2 reuses
   the phase 1 files, so differences between phase 1 and phase 2 results come
   from the design, not from revisions to Yahoo's adjusted prices.
3. FRED series: OECD short rates (carry and historical FX), SONIA (sterling
   splice), constant-maturity yields, the 3-month bill and daily FX spot.
4. Kenneth French daily factors (US market excess return, historical sample).
"""

import io
import re
import urllib.request
import zipfile

import pandas as pd
import requests
import yfinance as yf

from config import (DOWNLOAD_START, FX_RATE_SERIES, GBP_SPLICE_SERIES, HIST_BILL,
                    HIST_BOND_YIELDS, HIST_FX, HIST_FX_RATES, PHASE2_ADDITIONS,
                    PHASE2_CANDIDATES, PHASE2_LIQUIDITY_WINDOW, SAMPLE_END)
from data_io import HTTP_HEADERS, save_frame, with_retries
from fetch_prices import END_EXCLUSIVE, download_actions

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRENCH_DAILY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                    "F-F_Research_Data_Factors_daily_CSV.zip")
HIST_DOWNLOAD_START = "1985-01-01"


def screen_candidates() -> pd.DataFrame:
    rows = []
    lo, hi = PHASE2_LIQUIDITY_WINDOW
    for cls, tickers in PHASE2_CANDIDATES.items():
        for t in tickers:
            d = yf.download(t, start="1993-01-01", end=END_EXCLUSIVE, auto_adjust=False,
                            progress=False, multi_level_index=False)
            if d.empty:
                rows.append({"ticker": t, "class": cls, "first_trade": None,
                             "median_dollar_volume_m": None, "note": "no data (delisted)"})
                continue
            dv = (d["Close"] * d["Volume"]).loc[lo:hi]
            rows.append({"ticker": t, "class": cls, "first_trade": d.index[0].date(),
                         "median_dollar_volume_m": dv.median() / 1e6 if len(dv) else None,
                         "note": ""})
    return pd.DataFrame(rows).set_index("ticker")


def download_additions():
    tickers = list(PHASE2_ADDITIONS)
    raw = yf.download(tickers, start=DOWNLOAD_START, end=END_EXCLUSIVE, auto_adjust=False,
                      actions=False, progress=False, threads=False)
    missing = [t for t in tickers if raw["Close"][t].dropna().empty]
    if missing:
        raise RuntimeError(f"no price data for {missing}")
    return raw, tickers


def fred(series: str, start: str) -> pd.Series:
    raw = urllib.request.urlopen(FRED_URL.format(series), timeout=60).read()
    s = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")       # FRED marks holidays with "."
    s.index.name = "date"
    return s.loc[start:SAMPLE_END].rename(series)


def download_french_daily() -> pd.DataFrame:
    response = requests.get(FRENCH_DAILY_URL, headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")
    lines = [line for line in text.splitlines() if re.match(r"^\s*\d{8},", line)]
    df = pd.read_csv(io.StringIO("\n".join(["date,mkt_rf,smb,hml,rf"] + lines)))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df.set_index("date").loc[HIST_DOWNLOAD_START:SAMPLE_END]


def main():
    screen = with_retries(screen_candidates)
    save_frame(screen, "phase2_screen")
    print(screen.to_string())

    raw, tickers = with_retries(download_additions)
    for field, name in [("Close", "u2add_close"), ("Adj Close", "u2add_adj_close"),
                        ("Volume", "u2add_volume")]:
        panel = raw[field][tickers]
        panel.index.name = "date"
        save_frame(panel, name)
    rows = []
    for t in tickers:
        for date, row in with_retries(download_actions, t).iterrows():
            rows.append({"date": date, "ticker": t, "dividend": row.get("Dividends", 0.0),
                         "split": row.get("Stock Splits", 0.0)})
    actions = pd.DataFrame(rows).set_index("date").sort_index()
    save_frame(actions, "u2add_actions")
    print(f"Additions: {len(tickers)} tickers, {raw['Adj Close'].shape[0]} dates; "
          f"{(actions['dividend'] > 0).sum()} distributions, {(actions['split'] > 0).sum()} splits")

    monthly = sorted(set(FX_RATE_SERIES.values()) | set(HIST_FX_RATES.values()))
    rates = pd.concat([with_retries(fred, s, HIST_DOWNLOAD_START) for s in monthly], axis=1)
    save_frame(rates, "fred_monthly_rates")
    daily_ids = ([GBP_SPLICE_SERIES, HIST_BILL] + [v[0] for v in HIST_BOND_YIELDS.values()]
                 + [v[0] for v in HIST_FX.values()] + ["DEXUSEU"])
    daily = pd.concat([with_retries(fred, s, HIST_DOWNLOAD_START) for s in daily_ids], axis=1)
    save_frame(daily, "fred_daily")
    print(f"FRED monthly: {rates.shape}, daily: {daily.shape}")

    french = with_retries(download_french_daily)
    save_frame(french, "french_daily")
    print(f"French daily: {len(french)} days, {french.index[0].date()} to {french.index[-1].date()}")


if __name__ == "__main__":
    main()
