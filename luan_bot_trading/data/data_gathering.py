#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe
==========================================

Rules:
    - New ticker or gap >= 14 days: Fetch full history from Tiingo
    - Gap < 14 days: Use yfinance for the gap
    - When using Tiingo, always fetch full history (not partial date range)

Tiingo free tier: 50 requests/hour, 1000 requests/day, ~30 years history.

Usage:
    python data_gathering.py [--reset-offset]
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

yfinance = None
try:
    import yfinance as yf
    yfinance = yf
except ImportError:
    pass

import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

load_dotenv()
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")

DB_FILE = Path(__file__).parent / "db.h5"
TICKER_CACHE = Path(__file__).parent / "sp400_tickers.json"
OFFSET_FILE = Path(__file__).parent / "stock_offset.txt"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"
BATCH_SIZE = 45
WEEK_GAP_DAYS = 14

# Market benchmark: iShares Core S&P Mid-Cap ETF
IJH_TICKER = "IJH"
MACRO_GROUP = "macros"

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
# PART 2: RETRIEVE S&P 400 MID-CAP TICKERS
# ==============================================================================

def get_sp400_tickers(force_refresh: bool = False) -> list:
    REFRESH_DAYS = 75

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
# PART 3: DATA FETCHERS
# ==============================================================================

def fetch_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yfinance is None:
        return pd.DataFrame()

    try:
        stock = yf.Ticker(ticker)
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        df = stock.history(start=start, end=end_dt.strftime("%Y-%m-%d"))

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.replace(' ', '_') for c in df.columns]

        # Ensure timezone-naive
        try:
            df['Date'] = df['Date'].dt.tz_localize(None)
        except TypeError:
            pass  # Already timezone-naive

        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"      yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


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
    Update rules:
        - New DB or new ticker: fetch full history from Tiingo
        - Gap < 14 days: use yfinance for gap
        - Gap >= 14 days: fetch full history from Tiingo
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
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    next_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")

    if next_date > END_DATE:
        print(f"  {ticker}: Up to date.")
        return

    gap = (datetime.strptime(END_DATE, "%Y-%m-%d") - latest_date).days

    # Small gap: use yfinance
    if gap < WEEK_GAP_DAYS:
        print(f"  {ticker}: Gap {gap} days. Filling with yfinance...")
        yf_data = fetch_from_yfinance(ticker, next_date, END_DATE)
        if not yf_data.empty:
            store_data(ticker, yf_data, group=group)
            print(f"      yfinance: {len(yf_data)} rows")
            return
        else:
            print(f"  {ticker}: yfinance failed. Falling back to Tiingo.")

    # Large gap or yfinance failed: fetch full history from Tiingo
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
    print("  DATA GATHERING - S&P 400 Mid-Cap Universe")
    print("=" * 60)
    print(f"  History:  {HISTORY_YEARS} years")
    print(f"  Sources:  yfinance (gap < {WEEK_GAP_DAYS} days), Tiingo (full history)")
    print(f"  Market:   {IJH_TICKER} (fetched each run)")
    print(f"  Batch:    {BATCH_SIZE} tickers per run")
    print("=" * 60)

    if args.reset_offset:
        print("[INFO] --reset-offset specified.")
        save_offset(0)

    # Update market benchmark (IJH) using same yfinance/Tiingo logic
    print(f"\n[INFO] Updating market benchmark {IJH_TICKER}...")
    try:
        update_ticker(IJH_TICKER, group=MACRO_GROUP)
    except Exception as e:
        print(f"  [ERROR] Failed to update {IJH_TICKER}: {e}")

    all_tickers = get_sp400_tickers()
    n_total = len(all_tickers)
    print(f"\n[INFO] Universe: {n_total} tickers")

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
    print(f"  Load ETF with:  pd.read_hdf('db.h5', 'macros/IJH')")
    print("=" * 60)


if __name__ == "__main__":
    main()
