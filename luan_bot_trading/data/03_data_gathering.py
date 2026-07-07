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
`close/adj_close` is internally consistent with `/api/splits/`).

Local derivation (per row):
    adj_factor  = close / adjusted_close   # cumulative split+div factor
    adj_open    = open    / adj_factor     # = open    * adjusted_close / close
    adj_high    = high    / adj_factor
    adj_low     = low     / adj_factor
    adj_close   = adjusted_close
    adj_volume  = volume  * adj_factor      # split+dividend adjusted

Storage columns (matches prior Tiingo-era output for downstream feature
builder compatibility):
    Date, Open, High, Low, Close, Volume,            # raw
    Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume

EODHD subscription: 1000 req/min, 100000 req/day. Per-ticker cost = 1 call.
Full backfill of ~962 companies takes ~16 minutes single invocation.

Usage:
    python 03_data_gathering.py [--reset-offset] [--batch N]
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
        "earnings pipeline and a 1000x speed-up on full backfill."
    )

DB_FILE = Path(__file__).parent / "db.h5"
OFFSET_FILE = Path(__file__).parent / "stock_offset.txt"
COMPANIES_KEY = "/metadata/sp400_companies"

HISTORY_YEARS = 15
START_DATE = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

H5_GROUP = "sp400"
# EODHD: 1000/min ~= 16/sec. 0.05s sleep keeps us comfortably under that.
EODHD_INTER_CALL_DELAY = 0.05
# Batch size larger than before since EODHD limit is 1000/min vs Tiingo's 50/hr.
# Default 500 per run still finishes ~25 sec of clock-time, capped by 0.05s sleep
# at ~20 req/sec, so ~25 sec per 500 tickers.
DEFAULT_BATCH_SIZE = 500

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
# PART 3: DATA FETCHER (EODHD)
# ==============================================================================

def fetch_from_eodhd(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch EOD daily rows from EODHD and derive adjusted OHLC+Volume locally.

    EODHD returns: {date, open, high, low, close, adjusted_close, volume}
    We compute:    {Date, Open..Volume (raw), Adj_Open..Adj_Volume (derived)}

    Returns an empty DataFrame if EODHD responds with no rows / errors --
    the caller iterates to the next alias.
    """
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
        # 404 (ticker not found), 4xx (bad request), 5xx (server) -- try next alias
        return pd.DataFrame()

    try:
        data = response.json()
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # EODHD guarantees date + open + high + low + close + adjusted_close + volume columns
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if col not in df.columns:
            return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])  # already 'YYYY-MM-DD' strings

    # Build the local-derivation rows.
    df_clean = pd.DataFrame()
    df_clean["Date"] = df["date"]
    df_clean["Open"]  = df["open"].astype(float)
    df_clean["High"]  = df["high"].astype(float)
    df_clean["Low"]   = df["low"].astype(float)
    df_clean["Close"] = df["close"].astype(float)
    df_clean["Volume"] = df["volume"].astype(float)

    # Per-row split+dividend adjustment factor (validated by validate_eodhd_adjclose.py)
    # adj_factor = close / adjusted_close
    #   pre-split / pre-dividend rows: factor > 1 (raw is larger than adjusted)
    #   post-adjustment rows: factor ~= 1
    # Guard against zero / NaN adjusted_close to avoid div-by-zero corruption.
    adj_close = df["adjusted_close"].astype(float)
    adj_close_safe = adj_close.where(adj_close > 0)
    df_clean["adj_factor"] = (df_clean["Close"] / adj_close_safe).astype(float)

    # Adjusted OHLC (price) -- divide by adj_factor (= multiply by adj_close/close)
    df_clean["Adj_Open"]   = df_clean["Open"]   / df_clean["adj_factor"]
    df_clean["Adj_High"]   = df_clean["High"]   / df_clean["adj_factor"]
    df_clean["Adj_Low"]    = df_clean["Low"]    / df_clean["adj_factor"]
    df_clean["Adj_Close"]  = adj_close

    # Adjusted Volume -- multiply by adj_factor (= volume * close / adj_close)
    df_clean["Adj_Volume"] = df_clean["Volume"] * df_clean["adj_factor"]

    # Drop the intermediate column so final storage matches OUTPUT_COLUMNS
    df_clean = df_clean[OUTPUT_COLUMNS]

    # Ensure timezone-naive (EODHD dates are calendar days already, no tz)
    if df_clean["Date"].dt.tz is not None:
        df_clean["Date"] = df_clean["Date"].dt.tz_localize(None)

    # Drop rows with NaN in critical adjusted columns (shouldn't happen but
    # defensive: catches any row with adj_close=0 from EODHD's response).
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
        # 404 / empty / network error -> try next alias
        time.sleep(EODHD_INTER_CALL_DELAY)
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

    # Skip companies flagged as having no EODHD-available price data
    # (per Design 9b / company_merge_design.md)
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
# MAIN: BATCH PROCESSING (per company)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch S&P 400 Mid-Cap price data (per company, EODHD)")
    parser.add_argument("--reset-offset", action="store_true", help="Reset to beginning")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Companies per run")
    args = parser.parse_args()
    batch_size = args.batch

    print("=" * 60)
    print("  DATA GATHERING - S&P 400 Historical Universe (per-COMPANY, EODHD)")
    print("=" * 60)
    print(f"  History:  {HISTORY_YEARS} years ({START_DATE} .. {END_DATE})")
    print(f"  Source:   EODHD /api/eod (full history, alias fallback)")
    print(f"  Throttle: {EODHD_INTER_CALL_DELAY}s between calls (limit 1000/min)")
    print(f"  Batch:    {batch_size} companies per run")
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

    end_idx = min(offset + batch_size, n_total)
    batch = companies[offset:end_idx]

    if end_idx == n_total and len(batch) < batch_size:
        remaining = batch_size - len(batch)
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
        time.sleep(EODHD_INTER_CALL_DELAY)

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
