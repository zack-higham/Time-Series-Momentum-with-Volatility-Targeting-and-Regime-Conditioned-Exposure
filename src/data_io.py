"""Shared helpers for reading and writing the cached data files."""

import time

import pandas as pd

from config import DATA_DIR, OUTPUT_DIR

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126"}


def save_frame(df: pd.DataFrame, name: str, folder=DATA_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv")


def load_frame(name: str, folder=DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(folder / f"{name}.csv", index_col=0, parse_dates=True)


def save_output(df: pd.DataFrame, name: str) -> None:
    save_frame(df, name, OUTPUT_DIR)


def with_retries(func, *args, attempts: int = 3, pause: float = 5.0, **kwargs):
    """Call func, retrying on network errors (Yahoo and the data libraries time out)."""
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # network failures surface as many exception types
            if attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({type(exc).__name__}); retrying")
            time.sleep(pause)
