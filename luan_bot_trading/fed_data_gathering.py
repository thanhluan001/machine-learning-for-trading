"""
FRED Macro Data Fetcher
Direct CSV download from FRED public endpoint. No API key, no pandas_datareader.
Stored in db.h5 under /macros/fred_{metric_name}
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db.h5"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

METRICS = {
    "DFF": "fed_funds_rate",
    "T10Y2Y": "yield_curve_spread",
    "VIXCLS": "vix_close",
    "DCOILWTICO": "wti_oil",
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredseries.csv"


def fetch(series_id: str, start_date="2009-01-01", end_date=None) -> pd.DataFrame:
    if end_date is None:
        end_date = END_DATE
    url = f"{FRED_CSV_URL}?id={series_id}&cosd={start_date}&coed={end_date}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(r.content)
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def update(start_date="2009-01-01"):
    print(f"FRED update | {END_DATE}")
    for series_id, name in METRICS.items():
        path = f"/macros/fred_{name}"
        df = fetch(series_id, start_date=start_date)
        df.columns = ["Date", name]
        df.to_hdf(DB_PATH, key=path, mode="a", format="table", data_columns=["Date"])
        print(f"  {series_id}: {len(df)} rows")
        time.sleep(0.3)
    print("Done")


if __name__ == "__main__":
    update()
