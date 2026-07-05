#!/usr/bin/env python3
"""
Fetch Earnings Calendar (Proof of Concept)
==========================================
Fetches historical earnings calendar from yahooquery for all S&P 400 tickers.
Processes in batches of 20 to avoid rate limiting.

Usage:
    python fetch_earnings_poc.py [--reset-offset]
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from yahooquery import Ticker

# ==============================================================================
# CONFIG
# ==============================================================================

DB_FILE = Path(__file__).parent / "db.h5"
TICKER_CACHE = Path(__file__).parent / "sp400_tickers.json"
OFFSET_FILE = Path(__file__).parent / "earnings_offset.txt"

H5_EARNINGS = "earnings"
YEARS_HISTORY = 3
CUTOFF_DATE = datetime.now() - timedelta(days=YEARS_HISTORY * 365)

BATCH_SIZE = 20
SLEEP_BETWEEN = 1  # seconds between tickers to avoid IP ban


# ==============================================================================
# OFFSET TRACKING (same pattern as data_gathering.py)
# ==============================================================================

def get_offset() -> int:
    if OFFSET_FILE.exists():
        try:
            with open(OFFSET_FILE, "r") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return 0
    return 0


def save_offset(offset: int):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))


# ==============================================================================
# FETCHING
# ==============================================================================

def get_sp400_tickers() -> list:
    with open(TICKER_CACHE) as f:
        return json.load(f)


def fetch_earnings_single(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings from yahooquery.
    """
    try:
        t = Ticker(ticker)
        data = t.earnings

        if not data or ticker not in data:
            return pd.DataFrame()

        ed = data[ticker]

        if not isinstance(ed, dict):
            return pd.DataFrame()

        chart = ed.get("earningsChart", {})
        history = chart.get("quarterly", [])

        if not history:
            return pd.DataFrame()

        rows = []
        for item in history:
            reported_epoch = item.get("reportedDate")

            if not reported_epoch or reported_epoch == 0:
                continue

            dt = pd.to_datetime(reported_epoch, unit="s")

            if dt < pd.Timestamp(CUTOFF_DATE):
                continue

            rows.append({
                "Date": dt,
                "ticker": ticker,
                "eps_estimate": item.get("estimate"),
                "reported_eps": item.get("actual"),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return pd.DataFrame()


# ==============================================================================
# STORAGE
# ==============================================================================

def store_earnings(df: pd.DataFrame):
    """Append earnings DataFrame to HDF5 under /earnings/calendar. Deduplicates."""
    path = f"/{H5_EARNINGS}/calendar"

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if path in store.keys():
            existing = store[path]
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date", "ticker"])
            store.put(path, combined, format="table", data_columns=["Date", "ticker"])
        else:
            store.put(path, df, format="table", data_columns=["Date", "ticker"])


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch earnings calendar from Yahoo Finance")
    parser.add_argument("--reset-offset", action="store_true", help="Reset to beginning")
    args = parser.parse_args()

    if args.reset_offset:
        print("[INFO] --reset-offset specified.")
        save_offset(0)

    tickers = get_sp400_tickers()
    n_total = len(tickers)

    offset = get_offset()

    end_idx = min(offset + BATCH_SIZE, n_total)
    batch = tickers[offset:end_idx]

    print("=" * 60)
    print("  FETCH EARNINGS CALENDAR")
    print("=" * 60)
    print(f"  Source:    Yahoo Finance (via yahooquery)")
    print(f"  Tickers:   {n_total}")
    print(f"  Batch:     {BATCH_SIZE} per run (offset={offset})")
    print(f"  Progress:  {offset + 1}-{end_idx} of {n_total}")
    print("=" * 60)

    if not batch:
        # Wrap around when all tickers processed
        batch = tickers[:BATCH_SIZE]
        offset = 0
        end_idx = min(BATCH_SIZE, n_total)
        print("[INFO] Wrapped to start of ticker list.")

    success = 0
    empty = 0
    all_records = []

    for i, ticker in enumerate(batch):
        df = fetch_earnings_single(ticker)

        if not df.empty:
            success += 1
            all_records.append(df)
            print(f"  {ticker}: {len(df)} events")
        else:
            empty += 1
            print(f"  {ticker}: no data")

        if (i + 1) % 10 == 0 or (i + 1) == len(batch):
            print(f"      Progress: {i + 1}/{len(batch)} ({success} with data, {empty} empty)")

        time.sleep(SLEEP_BETWEEN)

    # Store batch results
    if all_records:
        batch_df = pd.concat(all_records, ignore_index=True)
        store_earnings(batch_df)
        print(f"\n[INFO] Stored {len(batch_df)} earnings records from this batch.")
    else:
        print("\n[INFO] No earnings data in this batch.")

    new_offset = end_idx if end_idx < n_total else 0
    save_offset(new_offset)
    print(f"[INFO] Saved offset: {new_offset} (run again for next batch)")

    print("\n" + "=" * 60)
    print(f"  Done. Database: {DB_FILE} under /{H5_EARNINGS}/calendar")
    print("=" * 60)


if __name__ == "__main__":
    main()
