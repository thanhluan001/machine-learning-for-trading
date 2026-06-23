"""
FRED Macro Data Fetcher
Uses fredapi (pip install fredapi).
Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db.h5"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
API_KEY = os.getenv("FRED_API_KEY")

METRICS = {
    "DFF": "fed_funds_rate",
    "T10Y2Y": "yield_curve_spread",
    "VIXCLS": "vix_close",
    "DCOILWTICO": "wti_oil",
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi",
}


def fetch(series_id: str, start_date="2009-01-01", end_date=None) -> pd.DataFrame:
    if end_date is None:
        end_date = END_DATE
    fred = Fred(api_key=API_KEY)
    s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = s.reset_index()
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def update(start_date="2009-01-01"):
    print(f"FRED update | {END_DATE}")
    for series_id, name in METRICS.items():
        path = f"/macros/fred_{name}"
        df = fetch(series_id, start_date=start_date)
        df.columns = ["Date", name]
        df.to_hdf(DB_PATH, key=path, mode="a", format="table", data_columns=["Date"] )
        print(f"  {series_id}: {len(df)} rows")
        time.sleep(0.3)
    print("Done")


if __name__ == "__main__":
    update()
