#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe (Company-Level, Tiingo-only)
=======================================================================

Fetches full Tiingo adjusted OHLCV history for every company that has ever
been a constituent of the S&P 400 (current + removed), per
/metadata/sp400_companies in db.h5. This avoids survivorship bias by including
delisted/removed names AND correctly handles ticker renames by anchoring on
the company-level canonical ticker (built by 02b_build_company_map.py).

Rules:
    - Iterate per **company** (not per ticker).
    - For each company, try the canonical ticker first, then the other aliases
      in priority order, on Tiingo. The first non-empty response is stored
      under /sp400/{canonical_ticker}; aliases are not stored individually.
    - Companies flagged `price_unavailable=True` are skipped + logged (no empty
      placeholder node is created).
    - Always fetch full history from Tiingo (no partial-range logic).
    - Uses adjusted close for split/dividend consistency.

Tiingo free tier: 50 requests/hour, 1000 requests/day, ~30 years history.

Usage:
    python 03_data_gathering.py [--reset-offset]
"""

import argparse
import json
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

# Explicit .env path so the script works regardless of CWD (matches 02b).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY not found. Please check your .env file.")

DB_FILE = Path(__file__).parent / "db.h5"
OFFSET_FILE = Path(__file__).parent / "stock_offset.txt"
COMPANIES_KEY = "/metadata/sp400_companies"

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
# PART 2: COMPANY UNIVERSE FROM /metadata/sp400_companies
# ==============================================================================

def get_all_companies() -> list[dict]:
    """Return the full list of company dicts from /metadata/sp400_companies.

    Each dict has keys: canonical_ticker, cik, aliases (list[str]),
                       name, sic, index_ref, combined_intervals,
                       per_ticker_intervals, price_unavailable
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01_metadata -> 02 -> 02b_build_company_map.py first."
        )
    # Detect the missing companies table and raise a helpful, named error so
    # the user knows which step is missing rather than getting a raw KeyError.
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if COMPANIES_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {COMPANIES_KEY} missing in {DB_FILE}. "
                f"Run 02b_build_company_map.py first (it produces /metadata/sp400_companies)."
            )
    df = pd.read_hdf(DB_FILE, key=COMPANIES_KEY)
    companies = []
    for _, row in df.iterrows():
        # Re-hydrate JSON list/dict columns
        try:
            aliases = json.loads(row["aliases"]) if isinstance(row["aliases"], str) else row["aliases"]
        except Exception:
            aliases = [row["canonical_ticker"]]
        if not aliases:
            aliases = [row["canonical_ticker"]]
        try:
            combined = json.loads(row["combined_intervals"]) if isinstance(row["combined_intervals"], str) else row["combined_intervals"]
        except Exception:
            combined = []
        companies.append({
            "canonical_ticker": str(row["canonical_ticker"]),
            "cik": None if pd.isna(row.get("cik")) else str(row["cik"]),
            "aliases": [str(a) for a in aliases],
            "name": None if pd.isna(row.get("name")) else str(row["name"]),
            "sic": None if pd.isna(row.get("sic")) else str(row["sic"]),
            "index_ref": None if pd.isna(row.get("index_ref")) else str(row["index_ref"]),
            "combined_intervals": combined,
            "price_unavailable": bool(row["price_unavailable"]),
        })
    return companies


# ==============================================================================
# PART 3: DATA FETCHER
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
        # Don't log here at high verbosity; callers decide based on whether any
        # alias succeeded.
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


def fetch_first_alias_from_tiingo(aliases: list[str]) -> tuple[str | None, pd.DataFrame]:
    """Try each alias in priority order; return (alias_used, df) of the first
    non-empty response. If none succeed, return (None, empty df).
    """
    for alias in aliases:
        data = fetch_from_tiingo(alias, START_DATE, END_DATE)
        if not data.empty:
            return alias, data
        # 404 / network error -> try next alias
    return None, pd.DataFrame()


# ==============================================================================
# PART 4: STORAGE HELPERS
# ==============================================================================

def store_data(canonical_ticker: str, data: pd.DataFrame, group: str = H5_GROUP):
    h5_path = f"/{group}/{canonical_ticker}"
    with pd.HDFStore(DB_FILE, mode="a") as store:
        # Overwrite only the target node (never mode='w' on the whole DB)
        if h5_path in store:
            store.remove(h5_path)
        store.put(h5_path, data, format='table', data_columns=["Date"])


def canonical_exists(canonical_ticker: str, group: str = H5_GROUP) -> bool:
    if not DB_FILE.exists():
        return False
    with pd.HDFStore(DB_FILE, mode="r") as store:
        return f"/{group}/{canonical_ticker}" in store.keys()


def get_latest_date(canonical_ticker: str, group: str = H5_GROUP) -> pd.Timestamp:
    with pd.HDFStore(DB_FILE, mode="r") as store:
        df = store[f"/{group}/{canonical_ticker}"]
        latest = df["Date"].max()
    if hasattr(latest, "tz") and latest.tz is not None:
        latest = latest.tz_localize(None)
    return latest


# ==============================================================================
# PART 5: UPDATE LOGIC (per company)
# ==============================================================================

def update_company(company: dict, group: str = H5_GROUP):
    """Fetch + store full Tiingo history for one company (canonical)
    using alias fallback. See module docstring for rules.
    """
    canonical = company["canonical_ticker"]
    aliases = company["aliases"] or [canonical]

    # Skip companies with no Tiingo-available alias (per Design 9b / company_merge_design.md)
    if company.get("price_unavailable"):
        print(f"  {canonical}: SKIP (price_unavailable=True). aliases={aliases}")
        return

    # Case 1: No database exists yet
    if not DB_FILE.exists():
        print(f"  {canonical}: Initial fetch (aliases: {aliases})...")
        alias_used, data = fetch_first_alias_from_tiingo(aliases)
        if not data.empty:
            with pd.HDFStore(DB_FILE, mode="w") as store:
                store.put(f"/{group}/{canonical}", data, format='table', data_columns=["Date"])
            print(f"      Stored {len(data)} rows via {alias_used}")
        else:
            print(f"      No data from any alias.")
        return

    # Case 2: New company not yet stored
    if not canonical_exists(canonical, group=group):
        print(f"  {canonical}: New company. Fetching (aliases: {aliases})...")
        alias_used, data = fetch_first_alias_from_tiingo(aliases)
        if not data.empty:
            store_data(canonical, data, group=group)
            print(f"      Stored {len(data)} rows via {alias_used}")
        else:
            print(f"      No data from any alias.")
        return

    # Case 3: Existing company - check freshness
    latest_date = get_latest_date(canonical, group=group)
    if latest_date.strftime("%Y-%m-%d") >= END_DATE:
        print(f"  {canonical}: Up to date.")
        return

    gap = (datetime.strptime(END_DATE, "%Y-%m-%d") - latest_date).days
    print(f"  {canonical}: Refetch full history (gap={gap}d, aliases: {aliases})...")
    alias_used, data = fetch_first_alias_from_tiingo(aliases)
    if not data.empty:
        store_data(canonical, data, group=group)
        print(f"      Stored {len(data)} rows via {alias_used}")


# ==============================================================================
# MAIN: BATCH PROCESSING (per company)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch S&P 400 Mid-Cap price data (per company)")
    parser.add_argument("--reset-offset", action="store_true", help="Reset to beginning")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Historical Universe (per-COMPANY)")
    print("=" * 60)
    print(f"  History:  {HISTORY_YEARS} years")
    print(f"  Source:   Tiingo (full history, alias fallback)")
    print(f"  Batch:    {BATCH_SIZE} companies per run")
    print("=" * 60)

    if args.reset_offset:
        print("[INFO] --reset-offset specified.")
        save_offset(0)

    companies = get_all_companies()
    n_total = len(companies)
    print(f"\n[INFO] Universe: {n_total} companies (from {COMPANIES_KEY})")

    offset = get_offset()
    print(f"[INFO] Current offset: {offset}")

    if offset >= n_total:
        print("[INFO] All companies processed. Wrapping to start.")
        offset = 0

    end_idx = min(offset + BATCH_SIZE, n_total)
    batch = companies[offset:end_idx]

    if end_idx == n_total and len(batch) < BATCH_SIZE:
        remaining = BATCH_SIZE - len(batch)
        batch += companies[:remaining]
        end_idx = remaining

    print(f"[INFO] Processing {len(batch)} companies: {offset+1} to {end_idx} of {n_total}")

    for i, company in enumerate(batch):
        try:
            update_company(company)
        except Exception as e:
            print(f"  [ERROR] Failed to update {company['canonical_ticker']}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == len(batch):
            print(f"      Progress: {i + 1}/{len(batch)}")
        time.sleep(1)

    new_offset = end_idx if end_idx < n_total else 0
    save_offset(new_offset)
    print(f"\n[INFO] Saved offset: {new_offset} (run again for next batch)")

    print("\n" + "=" * 60)
    print(f"  Done. Database: {DB_FILE}")
    if n_total > 0:
        print(f"  Load company with: pd.read_hdf('db.h5', 'sp400/{companies[0]['canonical_ticker']}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
