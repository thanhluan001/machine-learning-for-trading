#!/usr/bin/env python3
"""
Data Gathering - S&P 400 Mid-Cap Universe (Company-Level, EODHD)
=======================================================================

Fetches full EODHD adjusted OHLCV history for every company that has ever
been a constituent of the S&P 400 (current + removed), per
/metadata/sp400_companies in db.h5. This avoids survivorship bias by
including delisted/removed names AND correctly handles ticker renames by
anchoring on the company-level canonical ticker (built by
02b_build_company_map.py).

Rules:
    - Iterate per **company** (not per ticker).
    - For each company, try the canonical ticker first (as {ticker}.US on
      EODHD), then the other aliases in priority order. The first non-empty
      response is stored under /sp400/{canonical_ticker}; aliases are not
      stored individually.
    - Companies flagged `price_unavailable=True` are skipped + logged
      (no empty placeholder node is created).
    - Always fetch full 15-year history (no partial-range logic).

EODHD schema adaptation
------------------------
EODHD's /api/eod/{TICKER}.US endpoint returns rows shaped:
    {date, open, high, low, close, adjusted_close, volume}

It does NOT expose adj_open / adj_high / adj_low / adj_volume directly. We
derive all four locally via the close/adjusted_close ratio, which encodes the
cumulative split + dividend reinvestment factor. This is exactly how Tiingo
computes `adjVolume` (same convention). Validated empirically in
`validate_eodhd_adjclose.py` (7/7 probe tickers PASS -- EODHD
`close/adj_close` is internally consistent with `/api/splits/`). The derivation
is purely local: no extra split/dividend lookup credits are consumed.

Local derivation (per row):
    adj_factor  = close / adjusted_close   # cumulative split+div factor
    adj_open    = open    / adj_factor     # = open * adjusted_close / close
    adj_high    = high    / adj_factor
    adj_low     = low     / adj_factor
    adj_close   = adjusted_close
    adj_volume  = volume  * adj_factor      # split+dividend adjusted

Storage columns (matches prior Tiingo-era output for downstream feature
builder compatibility):
    Date, Open, High, Low, Close, Volume,            # raw
    Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume

EODHD subscription: effectively unlimited. No throttle, no batching, no
offset checkpoint. One invocation processes the full universe.

Usage:
    python 03_data_gathering.py
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
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError(
        "EODHD_API_KEY not found in .env. The new 03 fetches price history "
        "from EODHD (replacing Tiingo) for schema/stability alignment with the "
        "earnings pipeline."
    )

DB_FILE = Path(__file__).parent / "db.h5"
COMPANIES_KEY = "/metadata/sp400_companies"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"

# Output column order matches the prior Tiingo-era storage so downstream
# feature builder / discovery notebooks can read /sp400/{TICKER} unchanged.
# EODHD returns raw OHLC + adjusted_close + raw volume only; we derive the
# 4 adjusted columns locally (validated by validate_eodhd_adjclose.py).
OUTPUT_COLUMNS = [
    "Date",
    "Open", "High", "Low", "Close", "Volume",
    "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume",
]

# ==============================================================================
# PART 2: COMPANY UNIVERSE FROM /metadata/sp400_companies
# ==============================================================================
def get_all_companies() -> list[dict]:
    """Return the full list of company dicts from /metadata/sp400_companies."""
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"{DB_FILE} not found. Run 01_metadata -> 02 -> 02b_build_company_map.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if COMPANIES_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {COMPANIES_KEY} missing in {DB_FILE}. "
                "Run 02b_build_company_map.py first (it produces /metadata/sp400_companies)."
            )
        df = pd.read_hdf(DB_FILE, key=COMPANIES_KEY)

    companies = []
    for _, row in df.iterrows():
        try:
            aliases = json.loads(row["aliases"]) if isinstance(row["aliases"], str) else row["aliases"]
        except Exception:
            aliases = [row["canonical_ticker"]]
        if not aliases:
            aliases = [row["canonical_ticker"]]

        try:
            combined = (
                json.loads(row["combined_intervals"])
                if isinstance(row["combined_intervals"], str)
                else row["combined_intervals"]
            )
        except Exception:
            combined = []

        companies.append(
            {
                "canonical_ticker": str(row["canonical_ticker"]),
                "cik": None if pd.isna(row.get("cik")) else str(row["cik"]),
                "aliases": [str(a) for a in aliases],
                "name": None if pd.isna(row.get("name")) else str(row["name"]),
                "sic": None if pd.isna(row.get("sic")) else str(row["sic"]),
                "index_ref": None if pd.isna(row.get("index_ref")) else str(row["index_ref"]),
                "combined_intervals": combined,
                "price_unavailable": bool(row["price_unavailable"]),
            }
        )
    return companies

# ==============================================================================
# PART 3: DATA FETCHER (EODHD)
# ==============================================================================
def fetch_from_eodhd(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch EOD daily rows from EODHD and derive adjusted OHLC+Volume locally."""
    url = f"https://eodhd.com/api/eod/{ticker}.US"
    params = {
        "from": start,
        "to": end,
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "period": "d",
    }
    try:
        response = requests.get(url, params=params, timeout=60)
    except Exception:
        return pd.DataFrame()

    if response.status_code != 200:
        return pd.DataFrame()

    try:
        data = response.json()
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if col not in df.columns:
            return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])

    df_clean = pd.DataFrame()
    df_clean["Date"] = df["date"]
    df_clean["Open"] = df["open"].astype(float)
    df_clean["High"] = df["high"].astype(float)
    df_clean["Low"] = df["low"].astype(float)
    df_clean["Close"] = df["close"].astype(float)
    df_clean["Volume"] = df["volume"].astype(float)

    adj_close = df["adjusted_close"].astype(float)
    adj_close_safe = adj_close.where(adj_close > 0)
    df_clean["adj_factor"] = (df_clean["Close"] / adj_close_safe).astype(float)

    df_clean["Adj_Open"] = df_clean["Open"] / df_clean["adj_factor"]
    df_clean["Adj_High"] = df_clean["High"] / df_clean["adj_factor"]
    df_clean["Adj_Low"] = df_clean["Low"] / df_clean["adj_factor"]
    df_clean["Adj_Close"] = adj_close
    df_clean["Adj_Volume"] = df_clean["Volume"] * df_clean["adj_factor"]

    df_clean = df_clean[OUTPUT_COLUMNS]

    if df_clean["Date"].dt.tz is not None:
        df_clean["Date"] = df_clean["Date"].dt.tz_localize(None)

    df_clean = df_clean.dropna(subset=["Adj_Close", "Adj_Volume"]).reset_index(drop=True)
    return df_clean


def fetch_first_alias_from_eodhd(aliases: list[str]) -> tuple[str | None, pd.DataFrame]:
    """Try each alias in priority order; return (alias_used, df) of the first
    non-empty response. If none succeed, return (None, empty df).
    """
    for alias in aliases:
        data = fetch_from_eodhd(alias, START_DATE, END_DATE)
        if not data.empty:
            return alias, data
        # 404 / empty / network error -> try next alias (no throttle needed)
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
    """Fetch + store full EODHD history for one company (canonical)
    using alias fallback. See module docstring for rules.
    """
    canonical = company["canonical_ticker"]
    aliases = company["aliases"] or [canonical]

    if company.get("price_unavailable"):
        print(f"  {canonical}: SKIP (price_unavailable=True). aliases={aliases}")
        return

    # Case 1: No database exists yet -- create it (safe: DB_FILE does not exist)
    if not DB_FILE.exists():
        print(f"  {canonical}: Initial fetch (aliases: {aliases})...")
        alias_used, data = fetch_first_alias_from_eodhd(aliases)
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
        alias_used, data = fetch_first_alias_from_eodhd(aliases)
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
    alias_used, data = fetch_first_alias_from_eodhd(aliases)
    if not data.empty:
        store_data(canonical, data, group=group)
        print(f"      Stored {len(data)} rows via {alias_used}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Historical Universe (unlimited EODHD)")
    print("=" * 60)
    print(f"  History:   {HISTORY_YEARS} years ({START_DATE} .. {END_DATE})")
    print(f"  Source:    EODHD /api/eod (full history, alias fallback)")
    print(f"  Throttle:  none (unlimited subscription)")
    print("=" * 60)

    companies = get_all_companies()
    n_total = len(companies)
    print(f"\n[INFO] Universe: {n_total} companies (from {COMPANIES_KEY})")
    if n_total == 0:
        print("[INFO] Nothing to do.")
        return

    progress_every = max(1, n_total // 20)  # log ~20 progress steps
    t0 = time.time()
    done = skipped = failed = 0

    for i, company in enumerate(companies):
        canonical = company["canonical_ticker"]
        try:
            update_company(company)
            if company.get("price_unavailable"):
                skipped += 1
            else:
                done += 1
        except Exception as e:
            failed += 1
            print(f" [ERROR] Failed to update {canonical}: {e}")

        if (i + 1) % progress_every == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            print(
                f"\r[PROGRESS] {i + 1}/{n_total} companies  |  "
                f"stored={done}, skipped={skipped}, failed={failed}  |  "
                f"elapsed={elapsed:.1f}s",
                end="" if (i + 1) < n_total else "\n",
            )

    print()
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Database: {DB_FILE}")
    print(f"  Load a company with: pd.read_hdf('db.h5', 'sp400/{companies[0]['canonical_ticker']}')'")
    print("=" * 60)


if __name__ == "__main__":
    main()
