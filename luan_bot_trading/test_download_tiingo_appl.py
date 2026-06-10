#!/usr/bin/env python3
"""
Test Tiingo API - 15 Year History for AAPL
============================================
Verifies Tiingo free tier allows 15 years of historical data.
(Tiingo free tier supports up to ~30 years.)

Usage:
    python test_download_tiingo_appl.py

Output:
    aapl_test.h5 with columns: Date, AdjClose, Volume
"""

import os
from datetime import datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")


HISTORY_YEARS = 15


def fetch_prices(ticker: str = "AAPL") -> pd.DataFrame:
    """Fetch N years of adjClose from Tiingo."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_YEARS * 365)

    print(f"[INFO] Fetching {ticker}: {start_date.date()} to {end_date.date()}")
    print(f"       (~{HISTORY_YEARS} years)")

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "token": TIINGO_API_KEY,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    print(f"[INFO] Retrieved {len(data)} trading days")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df_clean = df[["date", "adjClose", "adjVolume"]].copy()
    df_clean.columns = ["Date", "AdjClose", "Volume"]

    return df_clean


if __name__ == "__main__":
    print("=" * 50)
    print("  TIINGO API TEST - 15 Year History")
    print("=" * 50)

    df = fetch_prices("AAPL")

    print(f"\n[Preview] Shape: {df.shape}")
    print(f"          Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"          AdjClose range: ${df['AdjClose'].min():.2f} - ${df['AdjClose'].max():.2f}")

    output_file = "aapl_test.h5"
    with pd.HDFStore(output_file, mode="w") as store:
        store.put("aapl", df, format="table", data_columns=True)

    print(f"\n[Done] Saved to: {output_file}")
    print("       Load with: pd.read_hdf('aapl_test.h5', 'aapl')")
