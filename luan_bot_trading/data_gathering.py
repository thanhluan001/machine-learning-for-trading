#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe
==========================================
Maintains an HDF5 database of adjusted OHLCV prices for all S&P 400 Mid-Cap
companies. Fetches 15 years of history incrementally, detecting corporate
actions that require full history refresh.

Runs in BATCHES of 45 requests per invocation. Controlled by offset.txt.
Run multiple times throughout the day to stay within Tiingo's rate limits.

Tiingo free tier: 50 requests/hour, 1000 requests/day, ~30 years history.

Usage:
    python data_gathering.py [--reset-offset]

Output Files:
    db.h5               (HDF5 file with group /sp400 for equities)
    sp400_tickers.json  (cached list of S&P 400 tickers)
    offset.txt          (tracks position in ticker list across runs)

Stored columns per ticker (all adjusted for splits/dividends):
    Date, Open, High, Low, Close, Volume
"""

import os
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")

DB_FILE = Path("db.h5")
TICKER_CACHE = Path("sp400_tickers.json")
OFFSET_FILE = Path("offset.txt")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")

H5_GROUP = "sp400"
BATCH_SIZE = 45

# All adjusted columns from Tiingo (splits/dividends already applied)
ADJ_COLUMNS = [
    "date",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "adjVolume",
]
OUTPUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ==============================================================================
# PART 1: OFFSET TRACKING
# ==============================================================================


def get_offset() -> int:
    """Read current offset from file. Returns 0 if file doesn't exist."""
    if OFFSET_FILE.exists():
        with open(OFFSET_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_offset(offset: int):
    """Save current offset to file."""
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))


# ==============================================================================
# PART 2: RETRIEVE S&P 400 MID-CAP TICKERS
# ==============================================================================


def get_sp400_tickers(force_refresh: bool = False) -> list:
    """Fetch current S&P 400 Mid-Cap constituents from Wikipedia."""
    REFRESH_DAYS = 75  # ~2.5 months, less than quarterly

    if not force_refresh and TICKER_CACHE.exists():
        cache_age_days = (
            datetime.now() - datetime.fromtimestamp(TICKER_CACHE.stat().st_mtime)
        ).days
        if cache_age_days < REFRESH_DAYS:
            print(f"[INFO] Using cached S&P 400 tickers (age: {cache_age_days} days).")
            with open(TICKER_CACHE) as f:
                return json.load(f)
        else:
            print(f"[INFO] Cache is {cache_age_days} days old. Refreshing...")

    print("[INFO] Fetching S&P 400 Mid-Cap constituents from Wikipedia...")
    resp = requests.get(WIKI_URL, headers={'User-Agent': 'Mozilla/5.0'})
    resp.raise_for_status()
    tables = pd.read_html(resp.content)
    df = tables[0]
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    tickers = df['Symbol'].tolist()
    print(f"      Retrieved {len(tickers)} tickers.")

    with open(TICKER_CACHE, 'w') as f:
        json.dump(tickers, f, indent=2)

    return tickers


# ==============================================================================
# PART 3: INCREMENTAL DATA FETCHING WITH ADJUSTMENT DETECTION
# ==============================================================================


def _group_path(ticker: str) -> str:
    """Return the full HDF5 path for a ticker under its group."""
    return f"/{H5_GROUP}/{ticker}"


def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted OHLCV data from Tiingo."""
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

    return df_clean


def ticker_exists_in_db(ticker: str) -> bool:
    """Check if a ticker already has data in the database."""
    if not DB_FILE.exists():
        return False
    with pd.HDFStore(DB_FILE, mode="r") as store:
        return _group_path(ticker) in store.keys()


def get_latest_date(ticker: str) -> pd.Timestamp:
    """Get the most recent date for a ticker in the database."""
    with pd.HDFStore(DB_FILE, mode="r") as store:
        db_df = store[_group_path(ticker)]
        return db_df["Date"].max()


def store_data(ticker: str, data: pd.DataFrame, mode="a"):
    """Store data for a ticker. mode='w' for new file, 'a' for append."""
    h5_path = _group_path(ticker)
    # For initial write when no DB exists, use 'w'
    if mode == "w":
        with pd.HDFStore(DB_FILE, mode="w") as store:
            store.put(h5_path, data, format='table', data_columns=["Date"])
    else:
        with pd.HDFStore(DB_FILE, mode="a") as store:
            store.put(h5_path, data, format='table', data_columns=["Date"])


def append_data(ticker: str, new_data: pd.DataFrame):
    """Append new data to existing ticker table."""
    h5_path = _group_path(ticker)
    with pd.HDFStore(DB_FILE, mode="a") as store:
        store.append(h5_path, new_data)


def needs_full_refresh(ticker: str, latest_db_date: str) -> bool:
    """
    Check if the adjClose for the most recent DB date has changed on Tiingo.
    If yes, a corporate action occurred and the entire history was adjusted.
    """
    try:
        df_latest = fetch_ticker_data(ticker, latest_db_date, END_DATE)
        if df_latest.empty:
            return False

        db_date = pd.to_datetime(latest_db_date)
        match = df_latest[df_latest["Date"] == db_date]

        if match.empty:
            return True

        with pd.HDFStore(DB_FILE, mode="r") as store:
            db_df = store[_group_path(ticker)]
            db_row = db_df[db_df["Date"] == db_date]

            if db_row.empty:
                return True

            db_close = float(db_row.iloc[0]["Close"])
            tiingo_close = float(match.iloc[0]["Close"])

            if abs(db_close - tiingo_close) > 1e-6:
                print(f"    [ADJUSTMENT] {ticker}: adjClose changed for {latest_db_date}")
                print(f"      DB: {db_close:.6f}, Tiingo: {tiingo_close:.6f}")
                return True

            return False

    except Exception as e:
        print(f"      Warning: Could not verify {ticker}: {e}")
        return True


def update_ticker(ticker: str):
    """Update a single ticker. Incremental or full refresh if needed."""

    # Case 1: No database exists yet
    if not DB_FILE.exists():
        print(f"  {ticker}: Fetching full history...")
        data = fetch_ticker_data(ticker, START_DATE, END_DATE)
        if not data.empty:
            store_data(ticker, data, mode="w")
        return

    # Case 2: New ticker not yet in DB
    if not ticker_exists_in_db(ticker):
        print(f"  {ticker}: New ticker, fetching full history...")
        data = fetch_ticker_data(ticker, START_DATE, END_DATE)
        if not data.empty:
            store_data(ticker, data, mode="a")
        return

    # Case 3: Existing ticker - check if full refresh needed
    latest_date = get_latest_date(ticker)
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    if needs_full_refresh(ticker, latest_date_str):
        print(f"  {ticker}: Refetching full history (corporate action detected)...")
        data = fetch_ticker_data(ticker, START_DATE, END_DATE)
        if not data.empty:
            store_data(ticker, data, mode="a")
        return

    # Case 4: Incremental update
    next_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if next_date > END_DATE:
        print(f"  {ticker}: Up to date.")
        return

    print(f"  {ticker}: Incremental ({next_date} to {END_DATE})...")
    new_data = fetch_ticker_data(ticker, next_date, END_DATE)

    if new_data.empty:
        print(f"  {ticker}: No new data.")
        return

    append_data(ticker, new_data)

    print(f"  {ticker}: Added {len(new_data)} rows.")


# ==============================================================================
# MAIN: BATCH PROCESSING
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fetch S&P 400 Mid-Cap price data from Tiingo")
    parser.add_argument("--reset-offset", action="store_true", help="Reset offset to 0 and start from beginning")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Mid-Cap Universe")
    print("=" * 60)
    print(f"  History: {HISTORY_YEARS} years")
    print(f"  Stored:  Date, Open, High, Low, Close, Volume (all adj)")
    print(f"  Batch:   {BATCH_SIZE} requests per run")
    print("=" * 60)

    # Handle reset flag
    if args.reset_offset:
        print("[INFO] --reset-offset specified. Starting from beginning.")
        save_offset(0)

    # Get all tickers
    all_tickers = get_sp400_tickers()
    n_total = len(all_tickers)
    print(f"\n[INFO] Universe: {n_total} tickers")

    # Get current offset and determine batch
    offset = get_offset()
    print(f"[INFO] Current offset: {offset}")

    # Determine batch range (wrap around at end)
    if offset >= n_total:
        print("[INFO] All tickers processed. Wrapping to start for next cycle.")
        offset = 0

    end_idx = min(offset + BATCH_SIZE, n_total)
    batch = all_tickers[offset:end_idx]

    # Handle wrap-around case (offset + batch > total)
    if end_idx == n_total and len(batch) < BATCH_SIZE:
        remaining = BATCH_SIZE - len(batch)
        batch += all_tickers[:remaining]
        end_idx = remaining

    print(f"[INFO] Processing {len(batch)} tickers: {offset+1} to {end_idx} of {n_total}")

    # Process batch
    for i, ticker in enumerate(batch):
        try:
            update_ticker(ticker)
        except Exception as e:
            print(f"  [ERROR] Failed to update {ticker}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == len(batch):
            print(f"      Progress: {i + 1}/{len(batch)}")

    # Save offset for next run
    new_offset = end_idx if end_idx < n_total else 0
    save_offset(new_offset)
    print(f"\n[INFO] Saved offset: {new_offset} (run again later for next batch)")

    print("\n" + "=" * 60)
    print(f"  Done. Database: {DB_FILE}")
    print(f"  Load data with: pd.read_hdf('db.h5', 'sp400/TICKER')")
    print("=" * 60)


if __name__ == "__main__":
    main()
