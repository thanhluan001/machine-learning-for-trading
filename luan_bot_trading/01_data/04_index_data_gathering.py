#!/usr/bin/env python3
"""
Index Data Gathering - Sector ETFs + Market Indices
Fetches 15 years of OHLCV for: XLK, XLF, XLI, XLY, XLP, XLV, XLU, XLE, XLB,
XLRE, XLC, IJH, SPY, VIXY

Tiingo free tier: 50 requests/hour, 1000/day, ~30 years history.

Usage:
    python index_data_gathering.py
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")

DB_FILE = Path(__file__).parent / "db.h5"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

TARGET_GROUP = "macros"

TICKERS = [
    # iShares S&P Mid-Cap 400 style ETFs -- referenced in /metadata/sp400
    # `index_ref` via get_index_ref() in SEC_sector_gathering.py.
    "IJH",  # S&P MidCap 400 base (default style blend)
    "IJJ",  # S&P MidCap 400 Value
    "IJK",  # S&P MidCap 400 Growth
    "IJS",  # S&P SmallCap 600 Value (used for Energy bucket per SIC_code_to_index.md)
    # Select Sector SPDRs -- the sector buckets referenced in `index_ref`.
    "XLB",  # Materials
    "XLF",  # Financials
    "XLRE", # Real Estate
    "XLU",  # Utilities
    # Additional Select Sector SPDRs kept for breadth / cross-check
    "XLK", "XLI", "XLY", "XLP", "XLV", "XLE", "XLC",
    # Market / volatility benchmarks
    "SPY", "VIXY",
]

ADJ_COLUMNS = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
OUTPUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_from_tiingo(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start, "endDate": end, "token": TIINGO_API_KEY}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"      Error fetching {ticker}: {e}")
        return pd.DataFrame()

    if not data:
        print(f"      No data returned for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df_clean = df[ADJ_COLUMNS].copy()
    df_clean.columns = OUTPUT_COLUMNS

    if df_clean["Date"].dt.tz is not None:
        df_clean["Date"] = df_clean["Date"].dt.tz_localize(None)

    return df_clean


def store_data(ticker: str, data: pd.DataFrame):
    h5_path = f"/{TARGET_GROUP}/{ticker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        store.put(h5_path, data, format="table", data_columns=["Date"])


def update_ticker(ticker: str):
    h5_path = f"/{TARGET_GROUP}/{ticker}"

    # Check if already up to date
    if DB_FILE.exists():
        with pd.HDFStore(DB_FILE, mode="r") as store:
            if h5_path in store:
                latest = store[h5_path]["Date"].max()
                if hasattr(latest, "tz") and latest.tz is not None:
                    latest = latest.tz_localize(None)
                gap = (datetime.strptime(END_DATE, "%Y-%m-%d") - latest).days
                if gap <= 0:
                    print(f"  {ticker}: Up to date.")
                    return
                print(f"  {ticker}: Gap {gap} days. Fetching...")
            else:
                print(f"  {ticker}: New ticker. Fetching...")
    else:
        print(f"  {ticker}: Initial fetch...")

    data = fetch_from_tiingo(ticker, START_DATE, END_DATE)
    if data.empty:
        return

    if not DB_FILE.exists():
        with pd.HDFStore(DB_FILE, mode="w") as store:
            store.put(h5_path, data, format="table", data_columns=["Date"])
    else:
        store_data(ticker, data)

    print(f"      {len(data)} rows")


def main():
    print(f"Index Data Gathering | {len(TICKERS)} tickers")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("-" * 50)

    for ticker in TICKERS:
        try:
            update_ticker(ticker)
        except Exception as e:
            print(f"  [ERROR] Failed to update {ticker}: {e}")
        time.sleep(1)

    print("\nDone")


if __name__ == "__main__":
    main()
