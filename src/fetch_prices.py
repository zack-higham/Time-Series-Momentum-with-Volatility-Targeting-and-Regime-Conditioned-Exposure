"""Download daily ETF prices and corporate actions from Yahoo Finance.

Saves three price panels so the total-return adjustment can be audited:
  close      Yahoo "Close": split-adjusted, NOT adjusted for distributions
  adj_close  Yahoo "Adj Close": split- and distribution-adjusted (total return)
  volume     shares traded
plus every dividend and split per ticker. Strategy returns use adj_close;
check_data.py rebuilds the total return from close + dividends to verify it.
"""

import pandas as pd
import yfinance as yf

from config import DOWNLOAD_START, INSTRUMENTS, SAMPLE_END
from data_io import save_frame, with_retries

# yfinance treats `end` as exclusive, so ask for the day after the sample end.
END_EXCLUSIVE = (pd.Timestamp(SAMPLE_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def download_panel(tickers):
    raw = yf.download(tickers, start=DOWNLOAD_START, end=END_EXCLUSIVE,
                      auto_adjust=False, actions=False, progress=False, threads=False)
    missing = [t for t in tickers if raw["Close"][t].dropna().empty]
    if missing:
        raise RuntimeError(f"no price data for {missing}")
    return raw


def download_actions(ticker):
    actions = yf.Ticker(ticker).actions
    actions.index = actions.index.tz_localize(None).normalize()
    return actions.loc[DOWNLOAD_START:SAMPLE_END]


def main():
    tickers = list(INSTRUMENTS)
    raw = with_retries(download_panel, tickers)
    for field, name in [("Close", "close"), ("Adj Close", "adj_close"), ("Volume", "volume")]:
        panel = raw[field][tickers]
        panel.index.name = "date"
        save_frame(panel, name)

    rows = []
    for t in tickers:
        actions = with_retries(download_actions, t)
        for date, row in actions.iterrows():
            rows.append({"date": date, "ticker": t,
                         "dividend": row.get("Dividends", 0.0),
                         "split": row.get("Stock Splits", 0.0)})
    actions = pd.DataFrame(rows).set_index("date").sort_index()
    save_frame(actions, "actions")

    adj = raw["Adj Close"][tickers]
    print(f"Saved {adj.shape[0]} dates x {adj.shape[1]} tickers, "
          f"{adj.index[0].date()} to {adj.index[-1].date()}")
    print(f"Actions: {(actions['dividend'] > 0).sum()} dividends, "
          f"{(actions['split'] > 0).sum()} splits")


if __name__ == "__main__":
    main()
