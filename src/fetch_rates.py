"""Download the risk-free rate.

Primary: Yahoo ^IRX, the 13-week Treasury bill rate in annualised percent,
current to the sample end. Secondary: the Ken French MONTHLY RF series (one-month
T-bill return, percent per month), used only to cross-check ^IRX in check_data.py.
The French DAILY file is not used: it rounds RF to 0.01% per day, so it reads
zero throughout 2009-2017 and 5.1% in 2025 (true average ~4.2%).
"""

import io
import re
import zipfile

import pandas as pd
import requests
import yfinance as yf

from config import DOWNLOAD_START, RF_TICKER, SAMPLE_END
from data_io import HTTP_HEADERS, save_frame, with_retries
from fetch_prices import END_EXCLUSIVE

FRENCH_MONTHLY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                      "F-F_Research_Data_Factors_CSV.zip")


def download_irx():
    raw = yf.download(RF_TICKER, start=DOWNLOAD_START, end=END_EXCLUSIVE,
                      auto_adjust=False, progress=False)
    irx = raw["Close"][RF_TICKER].rename("irx_pct")
    irx.index.name = "date"
    return irx.to_frame()


def download_french_rf():
    response = requests.get(FRENCH_MONTHLY_URL, headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")
    # Monthly rows start with a 6-digit YYYYMM date; the annual section that
    # follows uses 4-digit years and is excluded.
    lines = [line for line in text.splitlines() if re.match(r"^\s*\d{6},", line)]
    header = "date,mkt_rf,smb,hml,rf"
    df = pd.read_csv(io.StringIO("\n".join([header] + lines)))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").loc[DOWNLOAD_START:SAMPLE_END]
    return df[["rf"]].rename(columns={"rf": "french_rf_pct_monthly"})


def main():
    irx = with_retries(download_irx)
    save_frame(irx, "irx")
    print(f"^IRX: {len(irx)} days, {irx.index[0].date()} to {irx.index[-1].date()}")

    french = with_retries(download_french_rf)
    save_frame(french, "french_rf")
    print(f"French RF: {len(french)} months, {french.index[0].date()} to {french.index[-1].date()}")


if __name__ == "__main__":
    main()
