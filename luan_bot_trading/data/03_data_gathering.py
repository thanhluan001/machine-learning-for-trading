#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe (Historical)
=======================================================

Fetches full Tiingo adjusted OHLCV history for every ticker that has ever
been a constituent of the S&P 400 (current + removed), per /metadata/sp400
in db.h5. This avoids survivorship bias by including delisted/removed names.

Rules:
    - Always fetch full history from Tiingo (no partial-range logic).
    - Uses adjusted close for split/dividend consistency.
    - Splits universe equally across batches via stock_offset.txt checkpoint.

Tiingo free tier: 50 requests/hour, 1000 requests/day, ~30 years history.

Usage:
    python data_gathering.py [--reset-offset]
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")

DB_FILE = Path(__file__).parent / "db.h5"
OFFSET_FILE = Path(__file__).parent / "stock_offset.txt"
METADATA_KEY = "/metadata/sp400"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"
BATCH_SIZE = 45

ADJ_COLUMNS = [
    "date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
]
OUTPUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ==============================================================================
# PART 1: OFFSET TRACKING
# ==============================================================================

def get_offset() -> int:
    if OFFSET_FILE.exists():
        with open(OFFSET_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_offset(offset: int):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))


# ==============================================================================
# PART 2: TICKER UNIVERSE FROM /metadata/sp400
# ==============================================================================

def get_all_tickers() -> list:
    """Return the full historical S&P 400 universe (current + removed) from
    /metadata/sp400 in db.h5. Includes survivorship-bias-safe removed names.
    """
    meta_df = pd.read_hdf(DB_FILE, key=METADATA_KEY)
    if "ticker" not in meta_df.columns:
        meta_df = meta_df.reset_index()
    tickers = meta_df["ticker"].astype(str).tolist()
    return tickers


# ==============================================================================
# PART 3: DATA FETCHERS
# ==============================================================================

def fetch_from_tiingo(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start,
        "endDate": end,
        "token": TIINGO_API_KEY
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"      Error fetching {ticker}: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df_clean = df[ADJ_COLUMNS].copy()
    df_clean.columns = OUTPUT_COLUMNS

    # Ensure timezone-naive
    if df_clean["Date"].dt.tz is not None:
        df_clean["Date"] = df_clean["Date"].dt.tz_localize(None)

    return df_clean


# ==============================================================================
# PART 4: STORAGE HELPERS
# ==============================================================================

def store_data(ticker: str, data: pd.DataFrame, group: str = H5_GROUP):
    h5_path = f"/{group}/{ticker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        store.put(h5_path, data, format='table', data_columns=["Date"])


def ticker_exists(ticker: str, group: str = H5_GROUP) -> bool:
    if not DB_FILE.exists():
        return False
    with pd.HDFStore(DB_FILE, mode="r") as store:
        return f"/{group}/{ticker}" in store.keys()


def get_latest_date(ticker: str, group: str = H5_GROUP) -> pd.Timestamp:
    with pd.HDFStore(DB_FILE, mode="r") as store:
        df = store[f"/{group}/{ticker}"]
        latest = df["Date"].max()
    if hasattr(latest, "tz") and latest.tz is not None:
        latest = latest.tz_localize(None)
    return latest


# ==============================================================================
# PART 5: UPDATE LOGIC
# ==============================================================================

def update_ticker(ticker: str, group: str = H5_GROUP):
    """
    Always fetch full history from Tiingo, regardless of gap size.
    Tiingo is the sole data source (no yfinance fallback).
    """

    # Case 1: No database exists yet
    if not DB_FILE.exists():
        print(f"  {ticker}: Initial fetch from Tiingo...")
        data = fetch_from_tiingo(ticker, START_DATE, END_DATE)
        if not data.empty:
            with pd.HDFStore(DB_FILE, mode="w") as store:
                store.put(f"/{group}/{ticker}", data, format='table', data_columns=["Date"])
        return

    # Case 2: New ticker not yet in DB
    if not ticker_exists(ticker, group=group):
        print(f"  {ticker}: New ticker. Fetching full history from Tiingo...")
        data = fetch_from_tiingo(ticker, START_DATE, END_DATE)
        if not data.empty:
            store_data(ticker, data, group=group)
        return

    # Case 3: Existing ticker - check gap
    latest_date = get_latest_date(ticker, group=group)
    if latest_date.strftime("%Y-%m-%d") >= END_DATE:
        print(f"  {ticker}: Up to date.")
        return

    gap = (datetime.strptime(END_DATE, "%Y-%m-%d") - latest_date).days
    print(f"  {ticker}: Fetching full history from Tiingo (gap={gap}d)...")
    data = fetch_from_tiingo(ticker, START_DATE, END_DATE)
    if not data.empty:
        store_data(ticker, data, group=group)
        print(f"      Tiingo: {len(data)} rows")


# ==============================================================================
# MAIN: BATCH PROCESSING
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch S&P 400 Mid-Cap price data")
    parser.add_argument("--reset-offset", action="store_true", help="Reset to beginning")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Historical Universe")
    print("=" * 60)
    print(f"  History:  {HISTORY_YEARS} years")
    print(f"  Source:   Tiingo (full history)")
    print(f"  Batch:    {BATCH_SIZE} tickers per run")
    print("=" * 60)

    if args.reset_offset:
        print("[INFO] --reset-offset specified.")
        save_offset(0)

    all_tickers = get_all_tickers()
    n_total = len(all_tickers)
    print(f"\n[INFO] Universe: {n_total} tickers (from {METADATA_KEY})")

    offset = get_offset()
    print(f"[INFO] Current offset: {offset}")

    if offset >= n_total:
        print("[INFO] All tickers processed. Wrapping to start.")
        offset = 0

    end_idx = min(offset + BATCH_SIZE, n_total)
    batch = all_tickers[offset:end_idx]

    if end_idx == n_total and len(batch) < BATCH_SIZE:
        remaining = BATCH_SIZE - len(batch)
        batch += all_tickers[:remaining]
        end_idx = remaining

    print(f"[INFO] Processing {len(batch)} tickers: {offset+1} to {end_idx} of {n_total}")

    for i, ticker in enumerate(batch):
        try:
            update_ticker(ticker)
        except Exception as e:
            print(f"  [ERROR] Failed to update {ticker}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == len(batch):
            print(f"      Progress: {i + 1}/{len(batch)}")
        time.sleep(1)

    new_offset = end_idx if end_idx < n_total else 0
    save_offset(new_offset)
    print(f"\n[INFO] Saved offset: {new_offset} (run again for next batch)")

    print("\n" + "=" * 60)
    print(f"  Done. Database: {DB_FILE}")
    print(f"  Load ticker with: pd.read_hdf('db.h5', 'sp400/{all_tickers[0]}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
