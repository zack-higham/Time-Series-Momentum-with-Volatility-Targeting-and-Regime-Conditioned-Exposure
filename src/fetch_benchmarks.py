"""Download the AQR Time Series Momentum factors (monthly excess returns).

AQR maintains an updated version of the Moskowitz, Ooi and Pedersen (2012)
futures-based TSMOM factor, in total and by asset class. It is an external
benchmark: if my ETF implementation captures the same strategy, the two return
series should be clearly positively correlated.
"""

import io

import pandas as pd
import requests

from config import SAMPLE_END
from data_io import HTTP_HEADERS, save_frame, with_retries

AQR_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
           "Time-Series-Momentum-Factors-Monthly.xlsx")


def download_aqr():
    response = requests.get(AQR_URL, headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()
    sheet = pd.read_excel(io.BytesIO(response.content), sheet_name="TSMOM Factors", header=None)
    # The table header is the row whose second cell reads "TSMOM".
    header_row = sheet.index[sheet.iloc[:, 1].astype(str).str.strip() == "TSMOM"][0]
    table = sheet.iloc[header_row + 1:, :6]
    table.columns = ["date", "tsmom", "tsmom_cm", "tsmom_eq", "tsmom_fi", "tsmom_fx"]
    table = table.dropna(subset=["date"])
    table["date"] = pd.to_datetime(table["date"])
    table = table.set_index("date").astype(float)
    # Label by calendar month-end so it aligns with my month-end index.
    table.index = table.index + pd.offsets.MonthEnd(0)
    return table.loc[:SAMPLE_END]


def main():
    aqr = with_retries(download_aqr)
    save_frame(aqr, "aqr_tsmom")
    print(f"AQR TSMOM: {len(aqr)} months, {aqr.index[0].date()} to {aqr.index[-1].date()}")


if __name__ == "__main__":
    main()
